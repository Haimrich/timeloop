/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <memory>

#include "model/util.hpp"
#include "model/level.hpp"
#include "model/buffer.hpp"
#include "pat/pat.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/network-legacy.hpp"
BOOST_CLASS_EXPORT(model::LegacyNetwork)

namespace model
{

LegacyNetwork::LegacyNetwork() // Need this to make Boost happy.
{ }

LegacyNetwork::LegacyNetwork(const Specs& specs) :
    specs_(specs)
{
  is_specced_ = true;
  is_evaluated_ = false;
}

LegacyNetwork::~LegacyNetwork()
{ }

LegacyNetwork::Specs LegacyNetwork::ParseSpecs(config::CompoundConfigNode network, std::size_t n_elements)
{
  (void) n_elements; // FIXME.

  Specs specs;

  // Network Type.
  specs.type = "Legacy";
  std::string name;
  network.lookupValue("name", name);
  specs.name = name;

  if (network.exists("attributes"))
  {
    network = network.lookup("attributes");
  }

  std::string legacy_subtype;
  if (network.lookupValue("network-type", legacy_subtype))
  {
    if (legacy_subtype.compare("1:1") == 0)
      specs.legacy_subtype = "1_1";
    else if (legacy_subtype.compare("1:N") == 0)
      specs.legacy_subtype = "1_N";
    else if (legacy_subtype.compare("M:N") == 0)
      specs.legacy_subtype = "M_N";
    else
    {
      std::cerr << "ERROR: Unrecognized legacy network subtype: " << legacy_subtype << std::endl;
      exit(1);
    }
  }
    
  // Word Bits.
  std::uint32_t word_bits;
  if (network.lookupValue("network-word-bits", word_bits))
  {
    specs.word_bits = word_bits;
  }
  else if (network.lookupValue("word-bits", word_bits) ||
           network.lookupValue("word_width", word_bits) ||
           network.lookupValue("datawidth", word_bits) )
  {
    // FIXME. Derive this from the buffer I'm connected to in the topology
    // instead of cheating and reading it directly from its specs config.
    specs.word_bits = word_bits;
  }
  else
  {
    specs.word_bits = Specs::kDefaultWordBits;
  }

  // Router energy.
  double router_energy;
  if (network.lookupValue("router-energy", router_energy)) {specs.router_energy = router_energy;}

  // Wire energy.
  double wire_energy;
  if (network.lookupValue("wire-energy", wire_energy)) {specs.wire_energy = wire_energy;}

  // Tile width.
  double tile_width;
  if (network.lookupValue("tile-width", tile_width)) {specs.tile_width = tile_width;}

  double energy_per_hop;
  if (network.lookupValue("energy-per-hop", energy_per_hop)) {
      specs.energy_per_hop = energy_per_hop;
  }

  // Latency specs
  std::string memory_interfaces_config;
  if (network.lookupValue("memory_interfaces", memory_interfaces_config)) {
      std::stringstream ss(memory_interfaces_config);
      std::uint64_t n;

      while(ss >> n) 
        specs.memory_interfaces.push_back(n);
  }
  double router_latency;
  if (network.lookupValue("router-latency", router_latency)) {
      specs.router_latency = router_latency;
  }
  double link_latency;
  if (network.lookupValue("link-latency", link_latency)) {
      specs.link_latency = link_latency;
  }

  return specs;
}

void LegacyNetwork::ConnectSource(std::weak_ptr<Level> source)
{
  source_ = source;
}

void LegacyNetwork::ConnectSink(std::weak_ptr<Level> sink)
{
  sink_ = sink;
}

void LegacyNetwork::SetName(std::string name)
{
  specs_.name = name;
}

std::string LegacyNetwork::Name() const
{
  return specs_.name;
}

void LegacyNetwork::AddConnectionType(ConnectionType ct)
{
  specs_.cType = static_cast<ConnectionType>(static_cast<int>(specs_.cType) | static_cast<int>(ct));
}

void LegacyNetwork::ResetConnectionType()
{
  specs_.cType = Unused;
}

bool LegacyNetwork::DistributedMulticastSupported() const
{
  bool retval = true;

  retval &= specs_.legacy_subtype == "M_N";

  return retval;
}

// Floorplanner interface.
void LegacyNetwork::SetTileWidth(double width_um)
{
  // Only set this if user didn't specify a pre-floorplanned tile width.
  specs_.tile_width = specs_.tile_width.IsSpecified() ? specs_.tile_width.Get() : width_um;
}

// Evaluate.
EvalStatus LegacyNetwork::Evaluate(const tiling::CompoundTile& tile,
                                 const bool break_on_failure)
{
  auto eval_status = ComputeAccesses(tile, break_on_failure);
  if (!break_on_failure || eval_status.success)
  {
    ComputeNetworkEnergy();
    ComputeSpatialReductionEnergy();
    ComputePerformance();
  }
  return eval_status;
}

EvalStatus LegacyNetwork::ComputeAccesses(const tiling::CompoundTile& tile, const bool break_on_failure)
{
  bool success = true;
  std::ostringstream fail_reason;

  //
  // 1. Collect stats (stats are always collected per-DataSpaceID).
  //
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
      
    stats_.utilized_instances[pv] = tile[pvi].replication_factor;      

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // Network access-count calculation for Read-Modify-Write datatypes depends on
      // whether the unit receiving a Read-Write datatype has the ability to do
      // a Read-Modify-Write (e.g. accumulate) locally. If the unit isn't capable
      // of doing this, we need to account for additional network traffic.

      // FIXME: need to account for the case when this level is bypassed. In this
      //        case we'll have to query a different level. Also size will be 0,
      //        we may have to maintain a network_size.

      // FIXME: perhaps this should be done during a tile post-processing phase instead
      //        of here.
      if (auto sink = sink_.lock())
      {      
        if (sink->HardwareReductionSupported() ||
            (specs_.cType == ConnectionType::ReadFill) )
        {
          stats_.ingresses[pv] = tile[pvi].accesses;
        }
        else
        {
          stats_.ingresses[pv].resize(tile[pvi].accesses.size());
          for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
          {
            if (tile[pvi].accesses[i] > 0)
            {
              // The following assertion is *incorrect* for coefficients (e.g. stride, pad) > 1.
              // FIXME: find a safety check that works with coefficients > 1.
              // assert(tile[pvi].size == 0 || tile[pvi].accesses[i] % tile[pvi].size == 0);
              stats_.ingresses[pv][i] = 2*tile[pvi].accesses[i] - tile[pvi].partition_size;
            }
            else
            {
              stats_.ingresses[pv][i] = 0;
            }
          }
        } // hardware reduction not supported

        sink.reset();
      }
      else
      {
        // Lock failed.
        std::cerr << "ERROR: attempt to access expired storage level." << std::endl;
        exit(1);
      }
    }
    else // Read-only data space.
    {
      stats_.ingresses[pv] = tile[pvi].accesses;
    }

    stats_.spatial_reductions[pv] = 0;
    stats_.distributed_multicast[pv] = tile[pvi].distributed_multicast;
    stats_.avg_hops[pv].resize(tile[pvi].accesses.size());
    for (unsigned i = 0; i < tile[pvi].accesses.size(); i++)
    {
      if (tile[pvi].accesses[i] > 0)
      {
        stats_.avg_hops[pv][i] = tile[pvi].cumulative_hops[i] / double(tile[pvi].scatter_factors[i]);
      }
    }
    
    // FIXME: issues with link-transfer modeling:
    // 1. link transfers should result in buffer accesses to a peer.
    // 2. should reductions via link transfers be counted as spatial or temporal?
    // poan: This two issues are fixed. For 1., see peer_accesses/fills stats in buffer.cpp
    // and for 2., we should count them as temporal reduction because the link
    // in this legacy/X-Y mesh network should not be able to reduce the partial
    // output.
    stats_.link_transfers[pv] = tile[pvi].link_transfers;

    stats_.fanout[pv] = tile[pvi].fanout;
    if (stats_.distributed_multicast.at(pv))
      stats_.distributed_fanout[pv] = tile[pvi].distributed_fanout;
    else
      stats_.distributed_fanout[pv] = 0;

    // FIXME: multicast factor can be heterogeneous. This is correctly
    // handled by energy calculations, but not correctly reported out
    // in the stats.
    stats_.multicast_factor[pv] = 0;

    for (unsigned i = 0; i < stats_.ingresses[pv].size(); i++)
    {
      if (stats_.ingresses[pv][i] > 0)
      {
        auto factor = i + 1;
        if (factor > stats_.multicast_factor[pv])
        {
          stats_.multicast_factor[pv] = factor;
        }
        if (problem::GetShape()->IsReadWriteDataSpace.at(pv) &&
                (specs_.cType & ConnectionType::UpdateDrain) )
        {
          stats_.spatial_reductions[pv] += (i * stats_.ingresses[pv][i]);
        }
      }
    }

  } // loop over pv.
    
    //
    // 2. Derive/validate architecture specs based on stats.
    //

    // Bandwidth constraints cannot be checked/inherited at this point
    // because the calculation is a little more involved. We will do
    // this later in the ComputePerformance() function.    

  (void) break_on_failure;
  is_evaluated_ = success;

  EvalStatus eval_status;
  eval_status.success = success;
  eval_status.fail_reason = fail_reason.str();
    
  return eval_status;

} // ComputeAccesses()

//
// Compute network energy.
//
void LegacyNetwork::ComputeNetworkEnergy()
{
#define PROBABILISTIC_MULTICAST 0
#define PRECISE_MULTICAST 1
#define EYERISS_HACK_MULTICAST 2  

#define MULTICAST_MODEL PROBABILISTIC_MULTICAST
  
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // WireEnergyPerHop checks if wire energy is 0.0 before using default pat
    double wire_energy = specs_.wire_energy.IsSpecified() ? specs_.wire_energy.Get() : 0.0;
    double energy_per_hop =
      specs_.energy_per_hop.IsSpecified() ?
      specs_.energy_per_hop.Get() : WireEnergyPerHop(specs_.word_bits.Get(), specs_.tile_width.Get(), wire_energy);
    double energy_per_router = specs_.router_energy.IsSpecified() ? specs_.router_energy.Get() : 0.0; // Set to 0 since no internal model yet
    
    auto fanout = stats_.distributed_multicast.at(pv) ?
      stats_.distributed_fanout.at(pv) :
      stats_.fanout.at(pv);

    double total_wire_hops = 0;
    std::uint64_t total_routers_touched = 0;
    double total_ingresses = 0;
    
    for (unsigned i = 0; i < stats_.ingresses[pv].size(); i++)
    {
      auto ingresses = stats_.ingresses.at(pv).at(i);
      total_ingresses += ingresses;
      if (ingresses > 0)
      {
        auto multicast_factor = i + 1;
#if MULTICAST_MODEL == PROBABILISTIC_MULTICAST

        auto num_hops = NumHops(multicast_factor, fanout);
        total_routers_touched += (1 + num_hops) * ingresses;

#elif MULTICAST_MODEL == PRECISE_MULTICAST

        (void)fanout;
        (void)multicast_factor;
        if (stats_.distributed_multicast.at(pv))
        {
          std::cerr << "ERROR: precise multicast calculation does not work with distributed multicast." << std::endl;
          exit(1);
        }
        auto num_hops = stats_.avg_hops.at(pv).at(i);
        total_routers_touched += (1 + std::uint64_t(std::floor(num_hops))) * ingresses;

#elif MULTICAST_MODEL == EYERISS_HACK_MULTICAST

        (void)fanout;
        unsigned num_hops = 0;
        
        // Weights are multicast, and energy is already captured in array access.
        // Assume weights are pv == 0.
        if (pv != 0)
        {
          // Input and Output activations are forwarded between neighboring PEs,
          // so the number of link transfers is equal to the multicast factor-1.
          num_hops = multicast_factor - 1;
        }
        
        // We pick up the router energy from the .cfg file as the "array" energy
        // as defined in the Eyeriss paper, so we don't add a 1 to the multicast
        // factor.
        total_routers_touched += num_hops * ingresses;

#else
#error undefined MULTICAST_MODEL
#endif        

        total_wire_hops += num_hops * ingresses;
      }
    }
    stats_.energy_per_hop[pv] = energy_per_hop;
    stats_.num_hops[pv] = total_ingresses > 0 ? total_wire_hops / total_ingresses : 0;
    stats_.energy[pv] =
      total_wire_hops * energy_per_hop + // wire energy
      total_routers_touched * energy_per_router; // router energy

    stats_.link_transfer_energy[pv] =
      stats_.link_transfers.at(pv) * (energy_per_hop + 2*energy_per_router);
  }
}

//
// Compute spatial reduction energy.
//
void LegacyNetwork::ComputeSpatialReductionEnergy()
{
  // Spatial reduction: add two values in the network.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv)
            && (specs_.cType & ConnectionType::UpdateDrain)) // also used for UpdateDrain connections
    {
      stats_.spatial_reduction_energy[pv] = stats_.spatial_reductions[pv] * 
        pat::AdderEnergy(specs_.word_bits.Get(), specs_.word_bits.Get());
    }
    else
    {
      stats_.spatial_reduction_energy[pv] = 0;
    }
  }    
}

void LegacyNetwork::ComputePerformance()
{
  stats_.cycles = 0;

  if (!specs_.router_latency.IsSpecified() || !specs_.link_latency.IsSpecified() || !specs_.memory_interfaces.size() ) return;

  auto source = source_.lock();
  auto sink = sink_.lock();
  if (!source || !sink) return;
  
  std::shared_ptr<BufferLevel> buffer_sink = std::dynamic_pointer_cast<BufferLevel>(sink);
  auto mapping_nest = buffer_sink->GetSubNest();
  auto read_buffer_specs = buffer_sink->GetSpecs();
  
  std::shared_ptr<BufferLevel> buffer_source = std::dynamic_pointer_cast<BufferLevel>(source);
  auto fill_buffer_specs = buffer_source->GetSpecs();

  std::uint64_t utilized_instances = stats_.fanout.at(0);
  std::uint64_t instances = fill_buffer_specs.instances.Get() / read_buffer_specs.instances.Get();
  std::uint64_t meshX = fill_buffer_specs.meshX.Get() / read_buffer_specs.meshX.Get();
  std::uint64_t meshY = instances / meshX;

  std::uint64_t compute_cycles = buffer_source->Cycles(); // FIXME

  if (!std::all_of(specs_.memory_interfaces.begin(), specs_.memory_interfaces.end(), [&](std::uint64_t p) { return p < instances; })) return; // FIXME

  int mapX = 1;
  int mapY = 1;
  for (auto loop = mapping_nest.rbegin(); loop != mapping_nest.rend(); loop++)
  {
    if (loop->spacetime_dimension == spacetime::Dimension::SpaceX) mapX *= loop->end;
    else if (loop->spacetime_dimension == spacetime::Dimension::SpaceY) mapY *= loop->end;
  }
  assert(mapX * mapY == (int)utilized_instances);
  stats_.mapX = (uint64_t)mapX;
  stats_.mapY = (uint64_t)mapY;

  source.reset();
  sink.reset();

  // Matrici traffico nei router e nei link
  std::vector<std::vector<std::uint64_t>> routers_ingresses(meshY, std::vector<std::uint64_t>(meshX, 0));
  std::vector<std::vector<std::uint64_t>> horizontal_links_ingresses(meshY, std::vector<std::uint64_t>(meshX - 1, 0));
  std::vector<std::vector<std::uint64_t>> vertical_links_ingresses(meshY - 1, std::vector<std::uint64_t>(meshX, 0));

  // Calcolo coordinate entry points
  std::vector<std::pair<int, int>> memory_interfaces;
  for (auto entry_point : specs_.memory_interfaces) {
    memory_interfaces.push_back(std::make_pair<int, int>(entry_point / meshX, entry_point % meshX));
  }

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
  
    for (unsigned i = 0; i < stats_.ingresses.at(pv).size(); i++)
    {
      if (stats_.ingresses[pv][i] > 0)
      {
        auto factor = i + 1;
        std::uint64_t ingresses_per_pe = factor * stats_.ingresses[pv][i] / utilized_instances;

        auto axis = spacetime::Dimension::Num;
        if (factor > 1) axis = FindMulticastAxis(pv, factor, mapping_nest);

        std::uint64_t last_distance, distance;
        std::pair<int, int> nearest_entry_point;
        for (int x = 0; x < mapX; x++)
          for (int y = 0; y < mapY; y++)
          {
            // Trova Entry Point più vicino
            last_distance = UINT64_MAX;
            for (unsigned epi = 0; epi < memory_interfaces.size(); epi++)
            {
              distance = std::abs(memory_interfaces[epi].first - y) + std::abs(memory_interfaces[epi].second - x);
              if (distance < last_distance)
              {
                last_distance = distance;
                nearest_entry_point.first = memory_interfaces[epi].first;
                nearest_entry_point.second = memory_interfaces[epi].second;
              }
            }

            // UNICAST - Aggiungi traffico su tutti gli elementi di passaggio (Routing XY)
            if (factor == 1) { 
              // Routers sulla riga di partenza (X)
              for (auto i = std::min<std::uint64_t>(x, nearest_entry_point.second); i <= std::max<std::uint64_t>(x, nearest_entry_point.second); i++)
                routers_ingresses[nearest_entry_point.first][i] += ingresses_per_pe;
              // Routers sulla colonna di arrivo (Y)
              for (auto j = std::min<std::uint64_t>(y, nearest_entry_point.first); j <= std::max<std::uint64_t>(y, nearest_entry_point.first); j++)
                if (j != std::uint64_t(nearest_entry_point.first)) // Brutta cosa ma altrimenti contiamo l'incrocio due volte
                  routers_ingresses[j][x] += ingresses_per_pe;
              // Links sulla riga di partenza (X)
              for (auto i = std::min<std::uint64_t>(x, nearest_entry_point.second); i < std::max<std::uint64_t>(x, nearest_entry_point.second); i++)
                horizontal_links_ingresses[nearest_entry_point.first][i] += ingresses_per_pe;
              // Links sulla colonna di arrivo (Y)
              for (auto j = std::min<std::uint64_t>(y, nearest_entry_point.first); j < std::max<std::uint64_t>(y, nearest_entry_point.first); j++)
                vertical_links_ingresses[j][x] += ingresses_per_pe;
            // MULTICAST
            } else {
              // MULTICAST ORIZZONTALE
              if (axis == spacetime::Dimension::SpaceX) {
                // Occupa il router e il link adiacente nella direzione orizzontale dell'entry point più vicino
                if (nearest_entry_point.second > x) {
                  routers_ingresses[y][x] += ingresses_per_pe;
                  horizontal_links_ingresses[y][x] += ingresses_per_pe;
                } else if (nearest_entry_point.second < x) {
                  routers_ingresses[y][x] += ingresses_per_pe;
                  horizontal_links_ingresses[y][x-1] += ingresses_per_pe;
                // Se abbiamo raggiunto la colonna dell'entry point occupo router e link nella sua colonna
                } else {
                  for (auto j = std::min<std::uint64_t>(y, nearest_entry_point.first); j <= std::max<std::uint64_t>(y, nearest_entry_point.first); j++)
                    routers_ingresses[j][x] += ingresses_per_pe;
                  for (auto j = std::min<std::uint64_t>(y, nearest_entry_point.first); j < std::max<std::uint64_t>(y, nearest_entry_point.first); j++)
                    vertical_links_ingresses[j][x] += ingresses_per_pe;
                }
              // MULTICAST VERTICALE
              } else if (axis == spacetime::Dimension::SpaceY) {
                // Occupa il router e il link adiacente nella direzione orizzontale dell'entry point più vicino
                if (nearest_entry_point.first > y) {
                  routers_ingresses[y][x] += ingresses_per_pe;
                  vertical_links_ingresses[y][x] += ingresses_per_pe;
                } else if (nearest_entry_point.first < y) {
                  routers_ingresses[y][x] += ingresses_per_pe;
                  vertical_links_ingresses[y-1][x] += ingresses_per_pe;
                // Se abbiamo raggiunto la riga dell'entry point occupo router e link nella sua riga
                } else {
                  for (auto j = std::min<std::uint64_t>(x, nearest_entry_point.second); j <= std::max<std::uint64_t>(x, nearest_entry_point.second); j++)
                    routers_ingresses[y][j] += ingresses_per_pe;
                  for (auto j = std::min<std::uint64_t>(x, nearest_entry_point.second); j < std::max<std::uint64_t>(x, nearest_entry_point.second); j++)
                    horizontal_links_ingresses[y][j] += ingresses_per_pe;
                }
              // BROADCAST (XY)
              } else {
                routers_ingresses[y][x] += ingresses_per_pe;
                // Occupa il router e il link adiacente nella direzione orizzontale dell'entry point più vicino
                if (nearest_entry_point.second > x) {
                  horizontal_links_ingresses[y][x] += ingresses_per_pe;
                } else if (nearest_entry_point.second < x) {
                  horizontal_links_ingresses[y][x-1] += ingresses_per_pe;
                // Se abbiamo raggiunto la colonna dell'entry point occupo router e link nella sua colonna
                } else {
                  if (nearest_entry_point.first > y) {
                    vertical_links_ingresses[y][x] += ingresses_per_pe;
                  } else if (nearest_entry_point.first < y) {
                    vertical_links_ingresses[y-1][x] += ingresses_per_pe;
                  }
                }
              }
            } 
          }
        break;
      }
    }
  }

  std::uint64_t i, j;
  // Trovo elementi più trafficati
  // Routers
  std::uint64_t router_max_ingresses = 0;
  for (i = 0; i < meshY; i++)
    for (j = 0; j < meshX; j++)
      if (routers_ingresses[i][j] > router_max_ingresses)
        router_max_ingresses = routers_ingresses[i][j];

  // Links
  std::uint64_t link_max_ingresses = 0;
  for (i = 0; i < meshY; i++)
    for (j = 0; j < meshX - 1; j++)
      if (horizontal_links_ingresses[i][j] > link_max_ingresses)
        link_max_ingresses = horizontal_links_ingresses[i][j];
  for (i = 0; i < meshY - 1; i++)
    for (j = 0; j < meshX; j++)
      if (vertical_links_ingresses[i][j] > link_max_ingresses)
        link_max_ingresses = vertical_links_ingresses[i][j];
      
  // Router latency
  double total_router_latency = router_max_ingresses * specs_.router_latency.Get();

  // Link latency
  double total_link_latency = link_max_ingresses * specs_.link_latency.Get();

  // NoC Latency
  stats_.cycles = std::max(total_router_latency, total_link_latency);
  stats_.throttling = (double)compute_cycles / stats_.cycles;
  stats_.meshX = meshX;
  stats_.meshY = meshY;
}

bool LegacyNetwork::IsDimensionProjectionOfDataspace(problem::Shape::DimensionID dimension_id, problem::Shape::DataSpaceID dataspace_id)
{
  auto shape = problem::GetShape();
  auto &proj1 = shape->Projections;

  for(auto &proj2 : proj1.at(dataspace_id))
    for (auto &proj3 : proj2) 
      if (proj3.second == dimension_id) 
        return true;

  return false;
}

spacetime::Dimension LegacyNetwork::FindMulticastAxis(problem::Shape::DataSpaceID dataspace_id, unsigned multicast_factor, std::vector<loop::Descriptor> mapping_nest)
{
  for (auto loop = mapping_nest.rbegin(); loop != mapping_nest.rend(); loop++)
    if (loop->spacetime_dimension != spacetime::Dimension::Time && loop->end == (int)multicast_factor && !IsDimensionProjectionOfDataspace(loop->dimension, dataspace_id) )
      return loop->spacetime_dimension;
  
  return spacetime::Dimension::Num;
}

// We need the following method so that the connected buffer can
// query it to find the cost of temporal reductions. Ugh.
std::uint64_t LegacyNetwork::WordBits() const
{
  assert(is_specced_);
  return specs_.word_bits.Get();
}

std::uint64_t LegacyNetwork::Cycles() const
{
  return stats_.cycles;
}

//
// Printers.
//
void LegacyNetwork::Print(std::ostream& out) const
{
  // Print network name.
  out << specs_.name << std::endl;  
  out << std::endl;

  std::string indent = "    ";

  out << indent << "SPECS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << indent << "Type            : " << specs_.type << std::endl;
  out << indent << indent << "Legacy sub-type : " << specs_.legacy_subtype << std::endl;
  out << indent << indent << "ConnectionType  : " << specs_.cType << std::endl;
  out << indent << indent << "Word bits       : " << specs_.word_bits << std::endl;
  out << indent << indent << "Router energy   : " << specs_.router_energy << " pJ" << std::endl;
  out << indent << indent << "Wire energy     : " << specs_.wire_energy << " pJ/b/mm" << std::endl;

  out << std::endl;

  if (specs_.router_latency.IsSpecified() && specs_.router_latency.IsSpecified() && specs_.memory_interfaces.size() != 0) {
    
    out << indent << "PERFORMANCE" << std::endl;
    out << indent << "-----" << std::endl;
    out << indent << indent << "Cycles: " << stats_.cycles << "" << std::endl;
    out << indent << indent << "Throttling: " << stats_.throttling << "" << std::endl;
    out << indent << indent << "MeshX: " << stats_.meshX << indent << indent << "MapX: " << stats_.mapX << "" << std::endl;
    out << indent << indent << "MeshY: " << stats_.meshY << indent << indent << "MapY: " << stats_.mapY << "" << std::endl;
    
    /*
    out << indent << indent << "Entry Points: " << specs_.entry_points.size() << " -> ";
    for (auto i: specs_.entry_points) out << i << ' ';
    out << std::endl << indent << indent << "Router Traffic" << std::endl;
    for (unsigned i = 0; i < stats_.routers_ingresses.size(); i++)
    {
        for (unsigned j = 0; j < stats_.routers_ingresses[i].size(); j++)
        {
            out << "\t\t" << stats_.routers_ingresses[i][j];
        }
        out << std::endl;
    }
    */
    out << std::endl;
  }

  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

    out << indent + indent << "Fanout                                  : "
        << stats_.fanout.at(pv) << std::endl;
    out << indent + indent << "Fanout (distributed)                    : "
        << stats_.distributed_fanout.at(pv) << std::endl;
    if (stats_.distributed_multicast.at(pv))
      out << indent + indent << "Multicast factor (distributed)          : ";
    else
      out << indent + indent << "Multicast factor                        : ";
    out << stats_.multicast_factor.at(pv) << std::endl;
      
    auto total_accesses =
      std::accumulate(stats_.ingresses.at(pv).begin(),
                      stats_.ingresses.at(pv).end(),
                      static_cast<std::uint64_t>(0));
    out << indent + indent << "Ingresses                               : " << total_accesses << std::endl;
    
    std::string mcast_type = "@multicast ";
    if (stats_.distributed_multicast.at(pv))
      mcast_type += "(distributed) ";
    for (std::uint64_t i = 0; i < stats_.ingresses.at(pv).size(); i++)
      if (stats_.ingresses.at(pv)[i] != 0)
        out << indent + indent + indent << mcast_type << i + 1 << ": "
            << stats_.ingresses.at(pv)[i] << std::endl;

    out << indent + indent << "Link transfers                          : "
        << stats_.link_transfers.at(pv) << std::endl;
    out << indent + indent << "Spatial reductions                      : "
        << stats_.spatial_reductions.at(pv) << std::endl;
    
    out << indent + indent << "Average number of hops                  : "
        << stats_.num_hops.at(pv) << std::endl;
    
    out << indent + indent << "Energy (per-hop)                        : "
        << stats_.energy_per_hop.at(pv)*1000 << " fJ" << std::endl;

    out << indent + indent << "Energy (per-instance)                   : "
        << stats_.energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Energy (total)                          : "
        << stats_.energy.at(pv) * stats_.utilized_instances.at(pv)
        << " pJ" << std::endl;
    out << indent + indent << "Link transfer energy (per-instance)     : "
        << stats_.link_transfer_energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Link transfer energy (total)            : "
        << stats_.link_transfer_energy.at(pv) * stats_.utilized_instances.at(pv)
        << " pJ" << std::endl;    
    out << indent + indent << "Spatial Reduction Energy (per-instance) : "
        << stats_.spatial_reduction_energy.at(pv) << " pJ" << std::endl;
    out << indent + indent << "Spatial Reduction Energy (total)        : "
        << stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv)
        << " pJ" << std::endl;
  }
  
  out << std::endl;
}

//
// PAT interface.
//
double LegacyNetwork::WireEnergyPerHop(std::uint64_t word_bits, const double hop_distance,
                                       double wire_energy_override)
{
  double hop_distance_mm = hop_distance / 1000;
  if (wire_energy_override != 0.0)
  {
    // Internal wire model using user-provided average wire-energy/b/mm.
    return word_bits * hop_distance_mm * wire_energy_override;
  }
  else
  {
    return pat::WireEnergy(word_bits, hop_distance_mm);
  }
}

double LegacyNetwork::NumHops(std::uint32_t multicast_factor, std::uint32_t fanout)
{
  // Assuming central/side entry point.
  double root_f = std::sqrt(multicast_factor);
  double root_n = std::sqrt(fanout);
  return (root_n*root_f) + 0.5*(root_n-root_f) - (root_n/root_f) + 0.5;
  // return (root_n*root_f);
}

//
// Accessors.
//

STAT_ACCESSOR(double, LegacyNetwork, NetworkEnergy,
              (stats_.link_transfer_energy.at(pv) + stats_.energy.at(pv)) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, LegacyNetwork, SpatialReductionEnergy,
              stats_.spatial_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
  
STAT_ACCESSOR(double, LegacyNetwork, Energy,
              NetworkEnergy(pv) +
              SpatialReductionEnergy(pv))

} // namespace model
