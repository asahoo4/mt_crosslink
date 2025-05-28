// @HEADER
// **********************************************************************************************************************
//
//                                          Sam: A Starter Application Using Mundy
//                                             Copyright 2025 Bryce Palmer
//
// Sam is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Sam is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

// C++ core
#include <cmath>
#include <iostream>

// External
#include <mpi.h>  // for MPI_Comm, MPI_Finalize, etc

// Openrand
#include <openrand/philox.h>  // for openrand::Philox

// Kokkos
#include <Kokkos_Core.hpp>
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>         // stk::mesh::EntityRank
#include <stk_topology/topology.hpp>       // stk::topology
#include <stk_util/ngp/NgpSpaces.hpp>      // stk::ngp::ExecSpace, stk::ngp::RangePolicy
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_geom/distance.hpp>         // for mundy::geom::distance
#include <mundy_geom/primitives.hpp>       // for mundy::geom::Spherocylinder
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>        // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/LinkData.hpp>         // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_copy

namespace mundy {

struct COORDS {};
struct VEL {};
struct FORCE {};

struct SPRING_CONSTANT {};
struct EQUILIBRIUM_DISTANCE {};

// Class functor to run harmonic bond interactions
// This is taken from will.cpp in mundy (effectively)...
class eval_harmonic_bond {
 public:
  eval_harmonic_bond() {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const auto& bonded_view) const {
    auto r1 = get<COORDS>(bonded_view, 0);
    auto r2 = get<COORDS>(bonded_view, 1);

    double k_linear = get<SPRING_CONSTANT>(bonded_view)[0];
    double r0 = get<EQUILIBRIUM_DISTANCE>(bonded_view)[0];

    std::cout << "r1: " << r1 << std::endl;
    std::cout << "r2: " << r2 << std::endl;
    std::cout << "k_linear: " << k_linear << std::endl;
    std::cout << "r0: " << r0 << std::endl;

    // Calculate the distance vector
    auto r21 = r2 - r1;
    auto r21_length = math::norm(r21);
    auto r21_hat = r21 / r21_length;
    // Calculate the force for a linear spring
    auto f1 = -k_linear * (r21_length - r0) * r21_hat;

    // Update the force
    // NOTE: Particle 2 above maps to particle 1 here, and Particle 1 above maps to 0.
    // NOTE: XXX: This should actually be wrong, as there is a race condition due to different threads acting on nodes 0
    // and 1 at the same time.
    // get<FORCE>(bonded_view, 0) -= f1;
    // get<FORCE>(bonded_view, 1) += f1;
    // NOTE: XXX: The correct way of accessing the nodes via atomics is...
    auto force1 = get<FORCE>(bonded_view, 0);
    auto force2 = get<FORCE>(bonded_view, 1);
    Kokkos::atomic_add(&force1[0], -f1[0]);
    Kokkos::atomic_add(&force1[1], -f1[1]);
    Kokkos::atomic_add(&force1[2], -f1[2]);
    Kokkos::atomic_add(&force2[0], f1[0]);
    Kokkos::atomic_add(&force2[1], f1[1]);
    Kokkos::atomic_add(&force2[2], f1[2]);
  }

  void apply_to(auto& bonded_agg, const stk::mesh::Selector& subset_selector) {
    static_assert(bonded_agg.topology() == stk::topology::BEAM_2, "BONDED must be beam 2 top.");

    auto ngp_bonded_agg = mesh::get_updated_ngp_aggregate(bonded_agg);
    ngp_bonded_agg.template sync_to_device<COORDS, FORCE, SPRING_CONSTANT, EQUILIBRIUM_DISTANCE>();
    ngp_bonded_agg.template for_each((*this) /*use my operator as a lambda*/, subset_selector);
    ngp_bonded_agg.template modify_on_device<FORCE>();
  }

  void apply_to(auto& bonded_agg) {
    static_assert(bonded_agg.topology() == stk::topology::BEAM_2, "BONDED must be beam 2 top.");

    auto ngp_bonded_agg = mesh::get_updated_ngp_aggregate(bonded_agg);
    ngp_bonded_agg.template sync_to_device<COORDS, FORCE, SPRING_CONSTANT, EQUILIBRIUM_DISTANCE>();
    ngp_bonded_agg.template for_each((*this) /*use my operator as a lambda*/);
    ngp_bonded_agg.template modify_on_device<FORCE>();
  }
};

void run_main() {
  // STK usings
  using stk::mesh::Entity;
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::Selector;
  using stk::topology::ELEM_RANK;
  using stk::topology::NODE_RANK;

  // Mundy things
  using mesh::BulkData;
  using mesh::DeclareEntitiesHelper;
  using mesh::FieldComponent;
  using mesh::LinkData;
  using mesh::LinkMetaData;
  using mesh::MeshBuilder;
  using mesh::MetaData;
  using mesh::QuaternionFieldComponent;
  using mesh::ScalarFieldComponent;
  using mesh::Vector3FieldComponent;

  // Setup the STK mesh (boiler plate)
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3).set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();
  meta_data_ptr->set_coordinate_field_name("COORDS");
  std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  MetaData& meta_data = *meta_data_ptr;
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link data (boilerplate)
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);

  // Declare Spheres
  Part& sphere_part = meta_data.declare_part_with_topology("SPHERES", stk::topology::PARTICLE);
  stk::io::put_io_part_attribute(sphere_part);

  // Declare bonded interactions as entities
  Part& bonded_part = meta_data.declare_part_with_topology("BONDED", stk::topology::BEAM_2);
  stk::io::put_io_part_attribute(bonded_part);

  // Declare Fields
  auto& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto& node_vel_field = meta_data.declare_field<double>(NODE_RANK, "VEL");
  auto& node_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");

  auto& elem_spring_constant_field = meta_data.declare_field<double>(ELEM_RANK, "SPRING_CONSTANT");
  auto& elem_equilibrium_distance_field = meta_data.declare_field<double>(ELEM_RANK, "EQUILIBRIUM_DISTANCE");

  // Put fields on mesh
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_vel_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, meta_data.universal_part(), 3, nullptr);

  stk::mesh::put_field_on_mesh(elem_spring_constant_field, bonded_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_equilibrium_distance_field, bonded_part, 1, nullptr);

  // Setup io on fields
  auto transient_role = Ioss::Field::TRANSIENT;
  auto vector_3d_io_type = stk::io::FieldOutputType::VECTOR_3D;
  stk::io::set_field_role(node_vel_field, transient_role);
  stk::io::set_field_output_type(node_vel_field, vector_3d_io_type);
  stk::io::set_field_role(node_force_field, transient_role);
  stk::io::set_field_output_type(node_force_field, vector_3d_io_type);

  stk::io::set_field_role(elem_spring_constant_field, transient_role);
  stk::io::set_field_output_type(elem_spring_constant_field, stk::io::FieldOutputType::SCALAR);

  // Commit the mesh
  meta_data.commit();

  // Build our accessors and aggregates
  auto node_coords_accessor = Vector3FieldComponent(node_coords_field);
  auto node_vel_accessor = Vector3FieldComponent(node_vel_field);
  auto node_force_accessor = Vector3FieldComponent(node_force_field);

  auto elem_spring_constant_accessor = ScalarFieldComponent(elem_spring_constant_field);
  auto elem_equilibrium_distance_accessor = ScalarFieldComponent(elem_equilibrium_distance_field);

  auto sphere_agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
                        .add_component<COORDS, NODE_RANK>(node_coords_accessor)
                        .add_component<VEL, NODE_RANK>(node_vel_accessor)
                        .add_component<FORCE, NODE_RANK>(node_force_accessor);

  auto bonded_agg = make_aggregate<stk::topology::BEAM_2>(bulk_data, bonded_part)
                        .add_component<COORDS, NODE_RANK>(node_coords_accessor)
                        .add_component<FORCE, NODE_RANK>(node_force_accessor)
                        .add_component<SPRING_CONSTANT, ELEM_RANK>(elem_spring_constant_accessor)
                        .add_component<EQUILIBRIUM_DISTANCE, ELEM_RANK>(elem_equilibrium_distance_accessor);

  // Create entities
  DeclareEntitiesHelper dec_helper;

  // Creating the two endpoint nodes in the elastic network
  dec_helper.create_node()
      .owning_proc(0)
      .id(1)
      .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_vel_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0});

  dec_helper.create_node()
      .owning_proc(0)
      .id(2)
      .add_field_data<double>(&node_coords_field, {1.1, 0.0, 0.0})
      .add_field_data<double>(&node_vel_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0});

  dec_helper.create_node()
      .owning_proc(0)
      .id(3)
      .add_field_data<double>(&node_coords_field, {2.1, 0.0, 0.0})
      .add_field_data<double>(&node_vel_field, {0.0, 0.0, 0.0})
      .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0});

  // Declare a bonded interaction between the nodes
  dec_helper.create_element()
      .owning_proc(0)
      .id(1)
      .topology(stk::topology::BEAM_2)
      .add_part(&bonded_part)
      .nodes({1, 2})
      .add_field_data<double>(&elem_spring_constant_field, {1.0})
      .add_field_data<double>(&elem_equilibrium_distance_field, {1.0});
  dec_helper.create_element()
      .owning_proc(0)
      .id(2)
      .topology(stk::topology::BEAM_2)
      .add_part(&bonded_part)
      .nodes({2, 3})
      .add_field_data<double>(&elem_spring_constant_field, {3.14})
      .add_field_data<double>(&elem_equilibrium_distance_field, {1.0});

  // Declare the entities
  dec_helper.check_consistency(bulk_data);
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // XXX Dump all of the mesh info
  //   stk::mesh::impl::dump_all_mesh_info(bulk_data, std::cout);

  // Test run of the evaluate functor
  eval_harmonic_bond().apply_to(bonded_agg);

  // XXX Dump all of the mesh info after calculating forces!
  stk::mesh::impl::dump_all_mesh_info(bulk_data, std::cout);
}

}  // namespace mundy

int main(int argc, char** argv) {
  // Initialize MPI
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  mundy::run_main();

  // Finalize MPI
  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}
