#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <math.h>

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xadapt.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
#include <random>
#include <iterator>

namespace py = pybind11;

struct MutexWatershed
{
    // A Watershed Segmentation Execution Engine
    // Given a ground truth segmentation the constrained Segmentation can be computed in parallel.

    MutexWatershed( xt::pytensor<int64_t, 1> image_shape, 
                    xt::pytensor<int64_t, 2> offsets,
                    uint64_t n_attractive_channels, 
                    xt::pytensor<uint64_t, 1> dam_stride )
        : image_shape_(image_shape)
        , offsets_(offsets)
        , n_attractive_channels_(n_attractive_channels)
        , dam_stride_(dam_stride)
    {
        n_dims_ = image_shape_.size();
        n_points_ = xt::prod(image_shape_)(0);
        n_directions_ = offsets_.shape()[0];
        action_counter_ = 0;

        strides_ = xt::zeros<int64_t>({n_directions_});
        for ( uint64_t d = 0; d < n_directions_; ++d )
        {
            int64_t this_stride = 0;
            int64_t total_stride = 1;
            for ( int64_t n = n_dims_-1; n >= 0; --n )
            {
                this_stride += offsets_(d, n) * total_stride;
                total_stride *= image_shape_(n);
            }
            strides_(d) = this_stride;
        }

        // reset all arrays to initial values
        clear_all();
        set_bounds();
    }

    // TODO: Could be made constant
    uint64_t n_dims_;
    uint64_t n_points_;
    uint64_t n_directions_;
    uint64_t n_attractive_channels_;

    xt::pytensor<int64_t, 1> image_shape_;
    xt::pytensor<int64_t, 2> offsets_;
    xt::pytensor<int64_t, 1> strides_;
    xt::pytensor<uint64_t,1> dam_stride_;
    xt::xarray<bool>         bounds_;

    // Pointers to union-find data structures
    // Can be switched between unconstrained (uc) and constrained (c) arrays
    xt::pytensor<uint64_t,1> * parent_;
    xt::pytensor<uint64_t,1> * rank_;
    xt::pytensor<uint64_t,1> * region_size_;
    xt::pytensor<int64_t, 1> * actions_;

    // Union-find and bookkeeping arrays
    xt::pytensor<uint64_t,1> uc_parent_;
    xt::pytensor<uint64_t,1> uc_rank_;
    xt::pytensor<uint64_t,1> uc_region_size_;

    // Bookkeeping variables
    uint64_t action_counter_;
    bool finished_;
    xt::pyarray<bool> seen_actions;
    
    // Constrained data
    xt::pytensor<uint64_t,1> c_parent_;
    xt::pytensor<uint64_t,1> c_rank_;
    xt::pytensor<uint64_t,1> c_region_size_;
    xt::pytensor<int64_t, 1> c_actions_;    
    xt::pytensor<int64_t, 1> uc_actions_;

    // One directional adjacency list
    // maps root node to random pixel in connected component.
    // This makes checking for an existing link more expensive,
    // but saves resources since it does not require relabeling on 
    // graph contraction.
    std::vector<std::vector<int64_t>> dam_graph_;
    bool active_constraints_;
    
    // Buffer for merged dams
    std::vector<int64_t> dam_buffer_;

    void clear_all()
    {
        set_uc();
        clear();
        c_clear();
    }

    void clear()
    {
        // Reset minimum spanning tree
        action_counter_ = 0;
        
        uc_parent_ = xt::arange(n_points_);
        uc_rank_ = xt::zeros<uint64_t>({n_points_});
        uc_region_size_ = xt::ones<uint64_t>({n_points_});

        c_region_size_ = xt::ones<uint64_t>({n_points_});
        uc_actions_ = xt::zeros<int64_t>({n_points_ * n_directions_});
        seen_actions = xt::zeros<bool>({n_points_ * n_directions_});
        finished_ = false;
    }

    void c_clear()
    {
        // Reset constrained minimum spanning tree
        c_parent_ = xt::arange(n_points_);
        c_rank_ = xt::zeros<uint64_t>({n_points_});
        c_actions_ = xt::zeros<int64_t>({n_points_ * n_directions_});

        region_size_ = &uc_region_size_;
        active_constraints_ = false;
        dam_graph_ = std::vector<std::vector<int64_t>>(n_points_);
    }

    void set_bounds()
    {
    	std::vector<uint64_t> s;
        for ( uint64_t n = 0; n < n_dims_; ++n )
           s.push_back(image_shape_(n));
        s.push_back(n_directions_);
        bounds_ = xt::zeros<bool>(s);

        if ( n_dims_ == 2 )
        {
            fast_2d_set_bounds();
        }
        else if ( n_dims_ == 3 )
        {
            fast_3d_set_bounds();
        }
        else
        {
            std::cout << "WARNING: fallback to slow bound computation because image dimensions " << n_dims_ << " != 2 or 3" << std::endl;
            bounds_.reshape({n_points_, n_directions_});
            slow_set_bounds();
            return;
        }
        bounds_.reshape({n_points_, n_directions_});
    }

    void fast_2d_set_bounds()
    {
        xt::view(bounds_, xt::range(0, image_shape_(0), int64_t(dam_stride_(0))),
                         xt::range(0, image_shape_(1), int64_t(dam_stride_(1))),
                         xt::all()) = 1;

        xt::view(bounds_, xt::all(),
                         xt::all(),
                         xt::range(0, n_attractive_channels_, int64_t(1))) = 1;

        for ( uint64_t d = 0; d < n_directions_; ++d )
        {
            auto o0 = offsets_(d, 0);
            if ( o0 > 0 )
                xt::view(bounds_, xt::range(image_shape_(0)-1, image_shape_(0)-o0-1, int64_t(-1)), xt::all(), d) = 0;
            else if ( o0 < 0 )
                xt::view(bounds_, xt::range(0., -o0, 1), xt::all(), d) = 0;            
            
            auto o1 = offsets_(d, 1);
            if ( o1 > 0 )
                xt::view(bounds_, xt::all(), xt::range(image_shape_(1)-1, image_shape_(1)-o1-1, int64_t(-1)), d) = 0;
            else if (o1 < 0)
                xt::view(bounds_, xt::all(), xt::range(0., -o1, 1), d) = 0;
        }
    }

    void fast_3d_set_bounds()
    {
        xt::view(bounds_, xt::range(0, image_shape_(0), int64_t(dam_stride_(0))),
                         xt::range(0, image_shape_(1), int64_t(dam_stride_(1))),
                         xt::range(0, image_shape_(2), int64_t(dam_stride_(2))),
                         xt::all()) = 1;

        xt::view(bounds_, xt::all(),
                         xt::all(),
                         xt::all(),
                         xt::range(0, n_attractive_channels_, int64_t(1))) = 1;

        for ( uint64_t d = 0; d < n_directions_; ++d )
        {
            auto o0 = offsets_(d, 0);
            if ( o0 > 0 )
                xt::view(bounds_, xt::range(image_shape_(0)-1, image_shape_(0)-o0-1, int64_t(-1)), xt::all(), xt::all(), d) = 0;
            else if ( o0 < 0 )
                xt::view(bounds_, xt::range(0., -o0, 1.), xt::all(), xt::all(), d) = 0;
        
            auto o1 = offsets_(d, 1);
            if ( o1 > 0 )
                xt::view(bounds_, xt::all(), xt::range(image_shape_(1)-1, image_shape_(1)-o1-1, int64_t(-1)), xt::all(), d) = 0;
            else if ( o1 < 0 )
                xt::view(bounds_, xt::all(), xt::range(0., -o1, 1.), xt::all(), d) = 0;
        
            auto o2 = offsets_(d, 2);
            if ( o2 > 0 )
                xt::view(bounds_, xt::all(), xt::all(), xt::range(image_shape_(2)-1, image_shape_(2)-o2-1, int64_t(-1)), d) = 0;
            else if ( o2 < 0 )
                xt::view(bounds_, xt::all(), xt::all(), xt::range(0., -o2, 1.), d) = 0;
        }
    }

    void slow_set_bounds()
    {
        for ( uint64_t i = 0; i < n_points_; ++i )
            for ( uint64_t d = 0; d < n_directions_; ++d )
                bounds_(i, d) = is_valid_edge(i, d);
    }

    void compute_randomized_bounds()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> rng;
        uint64_t stride_product = 1;
        for ( uint64_t n = 0; n < n_dims_; ++n )
            stride_product *= dam_stride_(n);

        // NOTE: we have to substract 1 from the node product, because
        // uniform_distributions range is inclusive on both sides
        rng = std::uniform_int_distribution<>(0, stride_product-1);

        // set dam stride to 1 temporarily since it is replaced by rng
        auto real_stride = dam_stride_;
        for ( auto& d: dam_stride_ )
            d = 1;

        for ( uint64_t i = 0; i < n_points_; ++i )
            for ( uint64_t d = n_attractive_channels_; d < n_directions_; ++d )
                if ( is_valid_edge(i, d) )
                    bounds_(i, d) = (rng(gen) == 0);
        dam_stride_ = real_stride;
    }

    uint64_t find( uint64_t i )
    {
        if ( (*parent_)(i) == i )
        {
            return i;
        }
        else
        {
            (*parent_)(i) = find((*parent_)(i));
            return (*parent_)(i);
        }
    }

    // alternative version using while-loop
    // uint64_t find(uint64_t id) {
    //     uint64_t n(id);
    //     while (n != (*parent_)(n)) {
    //         n = (*parent_)(n);
    //     }

    //     uint64_t i(id), x;
    //     while (n != i) {
    //         x = (*parent_)(id);
    //         (*parent_)(id) = n;
    //         i = x;
    //     }
    // }

    inline
    uint64_t _get_direction( uint64_t e )
    {
        return e / n_points_;
    }

    inline 
    uint64_t _get_position( uint64_t e )
    {
        return e % n_points_;
    }

    auto get_flat_label_projection()
    {
        //  This function generates an id-invariant projection of the current segmentation
        //  that can be used as an input for a neural network.

        // Set dam stride to 1 temporarily to avoid image artifacts.
        auto real_stride = dam_stride_;
        for ( auto& d : dam_stride_ ) d = 1;

        xt::pytensor<float,1> region_projection = 
            xt::zeros<float>({n_points_ * n_directions_});

        for ( uint64_t i = 0; i < n_points_; ++i )
        {
            uint64_t root_i = find(i);
            for ( uint64_t d = 0; d < n_directions_; ++d )
            {
                if ( check_bounds(i, d) )
                    if ( root_i == find(i + strides_(d)) )
                        region_projection(d*n_points_ + i) = 1.;
            }
        }
        dam_stride_ = real_stride;
        return region_projection;
    }

    void set_state( const MutexWatershed & ouf )
    {
        // Copies segmentation state from other MWS object
        n_dims_ = ouf.n_dims_;
        action_counter_ = ouf.action_counter_;
        strides_ = ouf.strides_;
        n_points_ = ouf.n_points_;
        
        uc_rank_ = ouf.uc_rank_;
        c_rank_ = ouf.c_rank_;
        // uc_rank_ = &rank_;

        uc_region_size_ = ouf.uc_region_size_;
        c_region_size_ = ouf.c_region_size_;
        
        uc_parent_ = ouf.uc_parent_;
        c_parent_ = ouf.c_parent_;
        // uc_parent_ = &parent;
        // region_gt_label = ouf.region_gt_label;
        
        image_shape_ = ouf.image_shape_;
        c_actions_ = ouf.c_actions_;
        uc_actions_ = ouf.uc_actions_;
        // actions_ = &uc_actions;
        seen_actions = ouf.seen_actions;
        dam_graph_ = ouf.dam_graph_;

        set_uc();
    }

    inline
    uint64_t is_valid_edge( uint64_t i, uint64_t k )
    {
        int64_t index = i;
        for ( int64_t n = n_dims_-1; n >= 0; --n )
        {
            if ( k >= n_attractive_channels_ )
                if ( index % int64_t(dam_stride_(n)) != 0 )
                    return false;
            if ( offsets_(k, n) != 0 )
            {
                int64_t coord = index % image_shape_(n);
                if ( offsets_(k, n) < 0 )
                {
                    if ( coord < -offsets_(k, n) )
                        return false;
                }
                else
                {
                    if ( coord >= image_shape_(n) - offsets_(k, n) )
                        return false;
                }
            }
            index /= image_shape_(n);
        }
        return true;
    }

    inline
    bool check_bounds( uint64_t i, uint64_t d )
    {
        return bounds_(i, d);
    }

    void set_c()
    {
        parent_ = &c_parent_;
        rank_ = &c_rank_;
        region_size_ = &c_region_size_;
        active_constraints_ = true;
    }

    void set_uc()
    {
        parent_ = &uc_parent_;
        rank_ = &uc_rank_;
        region_size_ = &uc_region_size_;
        active_constraints_ = false;
    }

    inline 
    void merge_dams( uint64_t root_from, uint64_t root_to )
    {
        if ( dam_graph_[root_from].empty() )
            return;

        if ( dam_graph_[root_to].empty() )
        {
            dam_graph_[root_to] = dam_graph_[root_from];
            dam_graph_[root_from].clear();  // @kisuk
            return;
        }

        dam_buffer_.clear();
        dam_buffer_.reserve(std::max(dam_graph_[root_from].size(), dam_graph_[root_to].size()));

        std::merge(dam_graph_[root_from].begin(), dam_graph_[root_from].end(),
                   dam_graph_[root_to].begin(), dam_graph_[root_to].end(), 
                   std::back_inserter(dam_buffer_));

        dam_graph_[root_to] = dam_buffer_;
        dam_graph_[root_from].clear();
    }

    inline
    bool is_dam_constrained( uint64_t root_i, uint64_t root_j )
    {
        auto de_i = dam_graph_[root_i].begin();
        auto de_j = dam_graph_[root_j].begin();

        while ( de_i != dam_graph_[root_i].end() && de_j != dam_graph_[root_j].end() )
        {
            if ( *de_i < *de_j )
            {
                ++de_i;
            }
            else
            {
                if ( !(*de_j < *de_i) )
                    return true;
                ++de_j;
            }
        }
        return false;
    }

    inline 
    bool merge_roots( uint64_t root_i, uint64_t root_j )
    {
        if ( !active_constraints_ )
        {
            if ( is_dam_constrained(root_i, root_j) ) 
                return false;
            merge_dams(root_i, root_j);
        }
        else
        {
            throw std::runtime_error("not implemented");
        }

        // merge regions
        (*parent_)(root_i) = root_j;
        (*region_size_)(root_j) += (*region_size_)(root_i);
        return true;
    }    

    inline 
    bool dam_constrained_merge( uint64_t i, uint64_t j )
    {
        uint64_t root_i = find(i);
        uint64_t root_j = find(j);
    
        if ( root_i != root_j )
        {
            if ((*rank_)(root_i) < (*rank_)(root_j))
            {
                return merge_roots(root_i, root_j);
            }
            else if ((*rank_)(root_i) > (*rank_)(root_j))
            {
                return merge_roots(root_j, root_i);
            }
            else
            {
                if ( merge_roots(root_i, root_j) )
                {
                    (*rank_)(root_j) += 1;
                    return true;
                }
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    inline 
    bool add_dam_edge( uint64_t i, uint64_t j, uint64_t dam_edge )
    {
        uint64_t root_i = find(i);
        uint64_t root_j = find(j);
        if ( root_i != root_j )
        {
            if ( !is_dam_constrained(root_i, root_j) )
            {
                dam_graph_[root_i].insert(std::upper_bound(dam_graph_[root_i].begin(), dam_graph_[root_i].end(), dam_edge), dam_edge);
                dam_graph_[root_j].insert(std::upper_bound(dam_graph_[root_j].begin(), dam_graph_[root_j].end(), dam_edge), dam_edge);
                return true;
            }
        }
        return false;
    }

    void repulsive_mst_cut( const xt::pyarray<long> & edge_list )
    {
        for ( auto& e : edge_list ) 
        {
            uint64_t i = _get_position(e);
            uint64_t d = _get_direction(e);
            if ( check_bounds(i, d) )
            {
                int64_t j = int64_t(i) + strides_(d);
                if ( d < n_attractive_channels_ ) // attractive
                    uc_actions_(e) = dam_constrained_merge(i, j);
                else // repulisve
                    uc_actions_(e) = add_dam_edge(i, j, e);
            }
        }
    }

    void repulsive_ucc_mst_cut( const xt::pyarray<long> & edge_list, 
                                uint64_t num_iterations )
    {
        finished_ = true;
        action_counter_ = 0;        
        for ( auto& e : edge_list )
        {
            if ( (num_iterations != 0) and (num_iterations <= action_counter_) )
                finished_ = false;

            if ( !seen_actions(e) )
            {
                seen_actions(e) = true;
                uint64_t i = _get_position(e);
                uint64_t d = _get_direction(e);
                if ( check_bounds(i, d) )
                {
                    int64_t j = int64_t(i) + strides_(d);
                    if ( d < n_attractive_channels_ ) // attractive
                    {
                        // unconstrained merge
                        set_uc();
                        bool merged = dam_constrained_merge(i, j);
                        if ( merged ) ++action_counter_;
                        uc_actions_(e) = merged;
                    }
                    else // repulsive
                    {
                        set_uc();
                        uc_actions_(e) = add_dam_edge(i, j, e);
                    }
                }
            }
        }
    }

    bool is_finished()
    {
        return finished_;
    }

    /////////////// get functions ////////////////////

    auto get_seen_actions()
    {
        return seen_actions;
    }

    auto get_flat_label_image_only_merged_pixels()
    {
        set_uc();
        xt::pyarray<uint64_t> label = xt::zeros<uint64_t>({n_points_});
        for ( uint64_t i = 0; i < n_points_; ++i )
        {
            uint64_t root_i = find(i);
            uint64_t size = (*region_size_)(root_i);
            label(i) = (size > 1) ? (root_i + 1) : 0; // @kisuk
        }
        return label;
    }

    auto get_flat_uc_label_image_only_merged_pixels()
    {
        set_uc();
        return get_flat_label_image_only_merged_pixels();
    }

    auto get_flat_c_label_image_only_merged_pixels()
    {
        set_c();
        auto lab_img = get_flat_label_image_only_merged_pixels();
        set_uc();
        return lab_img;
    }

    auto get_flat_label_image()
    {
        xt::pyarray<long> label = xt::zeros<long>({n_points_});
        for ( uint64_t i = 0; i < n_points_; ++i )
            label(i) = find(i) + 1;
        return label;
    }

    auto get_action_counter()
    {
        return action_counter_;
    }

    auto get_flat_c_label_image()
    {
        set_c();
        auto lab_img = get_flat_label_image();
        set_uc();
        return lab_img;
    }

    auto get_flat_uc_label_image()
    {
        set_uc();
        return get_flat_label_image();
    }

    auto get_flat_applied_action()
    {
        return actions_;
    }

    auto get_flat_applied_uc_actions()
    {
        return uc_actions_;
    }

    auto get_flat_applied_c_actions()
    {
        return c_actions_;
    }
};


// Python Module and Docstrings

PYBIND11_PLUGIN(mutex_watershed)
{
    xt::import_numpy();

    py::module m("mutex_watershed", R"docu(
        Mutex watershed

        .. currentmodule:: mutex_watershed

        .. autosummary:
           :toctree: _generate

        MutexWatershed
    )docu");

    py::class_<MutexWatershed>(m, "MutexWatershed")
        .def(py::init<xt::pytensor<int64_t,1> , xt::pytensor<int64_t,2> , uint64_t, xt::pytensor<uint64_t,1>>())
        .def("clear_all", &MutexWatershed::clear_all)
        .def("get_flat_label_image_only_merged_pixels", &MutexWatershed::get_flat_label_image_only_merged_pixels)
        .def("get_flat_uc_label_image_only_merged_pixels", &MutexWatershed::get_flat_uc_label_image_only_merged_pixels)
        .def("get_flat_c_label_image_only_merged_pixels", &MutexWatershed::get_flat_c_label_image_only_merged_pixels)
        .def("get_flat_label_image", &MutexWatershed::get_flat_label_image)
        .def("get_flat_uc_label_image", &MutexWatershed::get_flat_uc_label_image)
        .def("get_flat_c_label_image", &MutexWatershed::get_flat_c_label_image)
        .def("get_flat_label_projection", &MutexWatershed::get_flat_label_projection)
        .def("set_state",  &MutexWatershed::set_state)
        .def("set_bounds",  &MutexWatershed::set_bounds)
        .def("check_bounds",  &MutexWatershed::check_bounds)
        .def("compute_randomized_bounds",  &MutexWatershed::compute_randomized_bounds)
        .def("repulsive_mst_cut",  &MutexWatershed::repulsive_mst_cut)
        .def("repulsive_ucc_mst_cut",  &MutexWatershed::repulsive_ucc_mst_cut)
        .def("set_uc", &MutexWatershed::set_uc)
        .def("check_bounds", &MutexWatershed::check_bounds)
        .def("is_finished", &MutexWatershed::is_finished)
        .def("get_seen_actions", &MutexWatershed::get_seen_actions)

        .def("get_action_counter", &MutexWatershed::get_action_counter)
        .def("get_flat_applied_action", &MutexWatershed::get_flat_applied_action)
        .def("get_flat_applied_uc_actions", &MutexWatershed::get_flat_applied_uc_actions)
        .def("get_flat_applied_c_actions", &MutexWatershed::get_flat_applied_c_actions);

    return m.ptr();
}


