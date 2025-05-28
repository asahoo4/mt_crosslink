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
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info

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
void run_main() {
    // STK usings
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
  mesh_builder
      .set_spatial_dimension(3)  //
      .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition
                                       // to all fields are simple.
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

  // Declare Fields
  // Node fields
  auto &node_coords_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto &node_vel_field = meta_data.declare_field<double>(NODE_RANK, "VEL");
  // Put fields on mesh
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_vel_field, meta_data.universal_part(), 3, nullptr);

  // Setup io on fields
  auto transient_role = Ioss::Field::TRANSIENT;
  stk::io::set_field_role(node_vel_field, transient_role);

  auto scalar_io_type = stk::io::FieldOutputType::SCALAR;
  auto vector_3d_io_type = stk::io::FieldOutputType::VECTOR_3D;
  stk::io::set_field_output_type(node_vel_field, vector_3d_io_type);

  // Commit the mesh
  meta_data.commit();
  


  // Build our accessors and aggregates
  // Create accessors
  auto node_coords_accessor = Vector3FieldComponent(node_coords_field);
  auto node_vel_accessor = Vector3FieldComponent(node_vel_field);

  auto sphere_agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, sphere_part)
  .add_component<COORDS,NODE_RANK>(node_coords_accessor)
  .add_component<VEL,NODE_RANK>(node_vel_accessor);

  // Create entities
  DeclareEntitiesHelper dec_helper;

  std::vector<std::array<double,3>> positions = {
  {1.0,2.0,3.0},
  {4.0,5.0,6.0},
  {7.0,8.0,9.0}
};
std::vector<std::array<double,3>> velocities = {
  {-1.0,0.0,0.0},
  {-0.5,0.5,0.0},
  { 0.0,0.0,0.0}
};

for (size_t i = 0; i < positions.size(); ++i) {
  const auto& pos = positions[i];
  const auto& vel = velocities[i];
  std::vector<double> coords{ pos[0], pos[1], pos[2] };
  std::vector<double> vels  { vel[0], vel[1], vel[2] };

  dec_helper.create_node()
    .owning_proc(0)
    .id( static_cast<int>(i) + 1 )
    .add_field_data<double>(&node_coords_field, coords)
    .add_field_data<double>(&node_vel_field,    vels);
}



  // Declare the entities
  dec_helper.check_consistency(bulk_data);
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // XXX Dump all of the mesh info
  stk::mesh::impl::dump_all_mesh_info(bulk_data, std::cout);
}

} // namespace mundy

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