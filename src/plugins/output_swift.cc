// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn & Michael Buehlmann (this file)
//		 2026 by Matthieu Schaller (this file)
//		 2026 by Will Roper (this file)
//
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifdef USE_HDF5
#include "HDF_IO.hh"
#include <array>
#include <cfloat>
#include <cmath>
#include <output_plugin.hh>
#include <unistd.h> // for unlink
#include <vector>

/**
 * @brief Signed 64-bit type used for cell counts and particle offsets.
 *
 * This is deliberately long long, rather than int64_t: HDF_IO's GetDataType<>
 * dispatches on typeid and recognises long long, while int64_t is a different
 * type on some supported platforms.
 */
using swift_count_t = long long;
static_assert(sizeof(swift_count_t) == 8,
              "SWIFT cell counts need to be 64 bit");

/** @brief Contiguous section of a particle dataset written by one MPI rank. */
struct swift_write_range {
  hsize_t file_start; //!< First particle index in the global dataset.
  hsize_t length;     //!< Number of particles in the section.
};

/**
 * @brief Convert a seven-element particle-type array to an HDF5 attribute.
 *
 * @param a Values indexed by Gadget/SWIFT particle type.
 *
 * @return Copy of @p a in the container expected by HDF_IO.
 */
template <typename T> std::vector<T> from_7array(const std::array<T, 7> &a) {
  return std::vector<T>{{a[0], a[1], a[2], a[3], a[4], a[5], a[6]}};
}

/**
 * @brief Wrap a scalar in the one-element array used by SWIFT attributes.
 *
 * @param a Scalar attribute value.
 *
 * @return One-element vector containing @p a.
 */
template <typename T> std::vector<T> from_value(const T a) {
  return std::vector<T>{{a}};
}

/**
 * @brief Write one SWIFT-compatible HDF5 initial-conditions file.
 *
 * Particles are stored in SWIFT top-level-cell order. Cells follow SWIFT's
 * flattened (x, y, z) convention, with z varying fastest. Within each cell,
 * contributions are ordered by MPI rank and retain each rank's original local
 * particle order. /Cells records counts, offsets, bounds, and cell centres so
 * SWIFT and swiftsimio can restrict reads to spatial subregions.
 *
 * With parallel HDF5, every rank opens the shared file collectively and writes
 * a union of hyperslabs. With serial HDF5, ranks use the same layout but take
 * turns opening and writing the file. Rank zero creates datasets and metadata
 * in both modes.
 *
 * @tparam write_real_t Floating-point type stored in particle datasets.
 */
template <typename write_real_t>
class swift_output_plugin : public output_plugin {

protected:
  int num_ranks_; //!< Number of MPI ranks, or one without MPI.
  int this_rank_; //!< Rank in MPI_COMM_WORLD, or zero without MPI.
  int num_files_; //!< Number of output files; SWIFT output always uses one.

  real_t lunit_;   //!< Position conversion factor to Mpc.
  real_t vunit_;   //!< Velocity conversion factor to km/s.
  real_t munit_;   //!< Mass conversion factor to 1e10 solar masses.
  real_t boxsize_; //!< Comoving box length in Mpc/h from the configuration.
  real_t hubble_param_; //!< Dimensionless Hubble parameter.
  real_t astart_;       //!< Initial scale factor.
  real_t zstart_;       //!< Initial redshift.
  bool blongids_;       //!< Whether particle IDs are stored as 64-bit integers.
  bool bdobaryons_;     //!< Whether gas particle fields are required.

  size_t cdim_;      //!< Number of top-level cells along each dimension.
  size_t ncells_;    //!< Total number of top-level cells (cdim_^3).
  double box_out_;   //!< Output box length in Mpc.
  double cell_size_; //!< Top-level cell side length in Mpc.

  std::array<uint64_t, 7> npart_;      //!< Particle counts in this output file.
  std::array<uint32_t, 7> npartTotal_; //!< Low words of global particle counts.
  std::array<uint32_t, 7> npartTotalHighWord_; //!< High words of global counts.
  std::array<double, 7>
      mass_;     //!< Constant-mass table; zero means per-particle masses.
  double time_;  //!< Initial scale factor written to /Header.
  double ceint_; //!< Initial gas internal energy in output units.
  double h_;     //!< Initial gas smoothing length in output units.

public:
  /**
   * @brief Initialise units, cell geometry, and rank-independent metadata.
   *
   * Rank zero creates the file and writes /Units, /Code,
   * /ICs_parameters, and the species-independent part of /Cells.
   *
   * Configuration keys used from [output] are filename, UseLongids, and
   * optional cdim (default 32). cdim must be in [1, 895]; the upper bound
   * keeps both cdim^3 and 3 * cdim^3 within signed MPI count limits.
   *
   * @param cf Complete monofonIC configuration.
   * @param pcc Cosmology calculator owned by the IC generator.
   * @throws std::runtime_error If cell geometry or box size is invalid.
   */
  explicit swift_output_plugin(config_file &cf,
                               std::unique_ptr<cosmology::calculator> &pcc)
      : output_plugin(cf, pcc, "SWIFT") {

    // SWIFT uses a single file as IC */
    num_files_ = 1;
    this_rank_ = 0;
    num_ranks_ = 1;

    real_t astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
    const double rhoc = 27.7536609198; // in h^2 1e10 M_sol / Mpc^3 assuming
                                       // SWIFT's internal constants

    hubble_param_ = pcc->cosmo_param_["h"];
    zstart_ = cf_.get_value<double>("setup", "zstart");
    astart_ = 1.0 / (1.0 + zstart_);
    boxsize_ = cf_.get_value<double>("setup", "BoxLength");

    lunit_ = boxsize_ / hubble_param_; // final units will be in Mpc (without h)
    vunit_ = boxsize_;                 // final units will be in km/s
    munit_ = rhoc * std::pow(boxsize_, 3) /
             hubble_param_; // final units will be in 1e10 M_sol

    blongids_ = cf_.get_value_safe<bool>("output", "UseLongids", true);
    bdobaryons_ = cf_.get_value<bool>("setup", "DoBaryons");

    //... SWIFT top-level cell grid .......................................
    cdim_ = cf_.get_value_safe<size_t>("output", "cdim", 32);
    box_out_ = lunit_;

    if (cdim_ == 0)
      throw std::runtime_error(
          "SWIFT output: 'cdim' must be larger than zero.");
    // cdim^3 has to fit both the int32 'nr_cells' attribute and the int MPI
    // counts used for the (3*ncells long) cell bound reductions
    if (cdim_ > 895)
      throw std::runtime_error(
          "SWIFT output: 'cdim' must not exceed 895 (cell count overflow).");
    if (!std::isfinite(box_out_) || box_out_ <= 0.0)
      throw std::runtime_error(
          "SWIFT output: box size must be finite and positive.");

    ncells_ = cdim_ * cdim_ * cdim_;
    cell_size_ = box_out_ / double(cdim_);

    for (int i = 0; i < 7; ++i) {
      npart_[i] = 0;
      npartTotal_[i] = 0;
      npartTotalHighWord_[i] = 0;
      mass_[i] = 0.0;
    }

    time_ = astart;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &this_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks_);
#endif

    if (bdobaryons_) {

      const double gamma =
          cf_.get_value_safe<double>("cosmology", "gamma", 5.0 / 3.0);
      const double YHe = pcc_->cosmo_param_["YHe"];
      const double omegab = pcc_->cosmo_param_["Omega_b"];
      const double Tcmb0 = pcc_->cosmo_param_["Tcmb"];

      // compute gas internal energy
      const double npol = (fabs(1.0 - gamma) > 1e-7) ? 1.0 / (gamma - 1.) : 1.0;
      const double unitv = 1e5;
      const double adec =
          1.0 / (160. * std::pow(omegab * hubble_param_ * hubble_param_ / 0.022,
                                 2.0 / 5.0));
      const double Tini =
          astart_ < adec ? Tcmb0 / astart_ : Tcmb0 / astart_ / astart_ * adec;
      const double mu =
          (Tini > 1.e4) ? 4.0 / (8. - 5. * YHe) : 4.0 / (1. + 3. * (1. - YHe));
      ceint_ = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / unitv / unitv;

      music::ilog.Print("Swift : Calculated initial gas temperature: %.2f K/mu",
                        Tini / mu);
      music::ilog.Print("Swift : set initial internal energy to %.2e km^2/s^2",
                        ceint_);

      h_ = boxsize_ / hubble_param_ / cf_.get_value<double>("setup", "GridRes");
      music::ilog.Print("Swift : set initial smoothing length to mean "
                        "inter-part separation: %.2f Mpc",
                        h_);
    }

    music::ilog.Print(
        "Swift : sorting particles into a %zu^3 top-level cell grid", cdim_);

    // Only ranks 0 writes the header
    if (this_rank_ != 0)
      return;

    // delete output file if it exists
    unlink(fname_.c_str());

    // create output HDF5 file
    HDFCreateFile(fname_);

    // Write UNITS header using the physical constants assumed internally by
    // SWIFT
    HDFCreateGroup(fname_, "Units");
    // written as 1-element arrays, the shape SWIFT uses and swiftsimio expects
    HDFWriteGroupAttribute(
        fname_, "Units", "Unit mass in cgs (U_M)",
        from_value<double>(1.98841e43)); // 10^10 Msun in grams
    HDFWriteGroupAttribute(fname_, "Units", "Unit length in cgs (U_L)",
                           from_value<double>(3.08567758149e24)); // 1 Mpc in cm
    HDFWriteGroupAttribute(
        fname_, "Units", "Unit time in cgs (U_t)",
        from_value<double>(3.08567758149e19)); // so that unit vel is 1 km/s
    HDFWriteGroupAttribute(fname_, "Units", "Unit current in cgs (U_I)",
                           from_value<double>(1.0)); // 1 Ampere
    HDFWriteGroupAttribute(fname_, "Units", "Unit temperature in cgs (U_T)",
                           from_value<double>(1.0)); // 1 Kelvin

    // swiftsimio identifies SWIFT files through this group
    HDFCreateGroup(fname_, "Code");
    HDFWriteGroupAttribute(fname_, "Code", "Code", std::string("SWIFT"));

    // Write MUSIC configuration header
    int order = cf_.get_value<int>("setup", "LPTorder");
    std::string load = cf_.get_value<std::string>("setup", "ParticleLoad");
    std::string tf = cf_.get_value<std::string>("cosmology", "transfer");
    std::string cosmo_set =
        cf_.get_value<std::string>("cosmology", "ParameterSet");
    std::string rng = cf_.get_value<std::string>("random", "generator");
    int do_fixing = cf_.get_value<bool>("setup", "DoFixing");
    int do_invert = cf_.get_value<bool>("setup", "DoInversion");
    int do_baryons = cf_.get_value<bool>("setup", "DoBaryons");
    int do_baryonsVrel = cf_.get_value<bool>("setup", "DoBaryonVrel");
    int L = cf_.get_value<int>("setup", "GridRes");

    HDFCreateGroup(fname_, "ICs_parameters");
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Code",
                           std::string("MUSIC2 - monofonIC"));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Git Revision",
                           std::string(GIT_REV));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Git Tag",
                           std::string(GIT_TAG));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Git Branch",
                           std::string(GIT_BRANCH));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Precision",
                           std::string(CMAKE_PRECISION_STR));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Convolutions",
                           std::string(CMAKE_CONVOLVER_STR));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "PLT",
                           std::string(CMAKE_PLT_STR));
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "LPT Order", order);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Particle Load", load);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Transfer Function", tf);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Cosmology Parameter Set",
                           cosmo_set);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Random Generator", rng);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Mode Fixing", do_fixing);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Mode inversion",
                           do_invert);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Baryons", do_baryons);
    HDFWriteGroupAttribute(fname_, "ICs_parameters",
                           "Baryons Relative Velocity", do_baryonsVrel);
    HDFWriteGroupAttribute(fname_, "ICs_parameters", "Grid Resolution", L);

    if (tf == "CLASS") {
      double ztarget = cf_.get_value<double>("cosmology", "ztarget");
      HDFWriteGroupAttribute(fname_, "ICs_parameters", "Target Redshift",
                             ztarget);
    }
    if (rng == "PANPHASIA") {
      std::string desc = cf_.get_value<std::string>("random", "descriptor");
      HDFWriteGroupAttribute(fname_, "ICs_parameters", "Descriptor", desc);
    }

    this->write_common_cell_metadata();
  }

  /**
   * @brief Write the final SWIFT header after all species counts are known.
   *
   * The IC generator destroys output plugins before finalising MPI. Header
   * creation is skipped during exception unwinding to avoid presenting a
   * partially written file as complete.
   */
  ~swift_output_plugin() {
    if (!std::uncaught_exceptions()) {
      if (this_rank_ == 0) {

        // Write Standard Gadget / SWIFT hdf5 header
        HDFCreateGroup(fname_, "Header");
        HDFWriteGroupAttribute(fname_, "Header", "Dimension",
                               from_value<int>(3));
        HDFWriteGroupAttribute(
            fname_, "Header", "BoxSize",
            std::vector<double>(3,
                                boxsize_ / hubble_param_)); // in Mpc, not Mpc/h
        HDFWriteGroupAttribute(fname_, "Header", "Scale-factor",
                               from_value<double>(time_));

        HDFWriteGroupAttribute(fname_, "Header", "NumPart_Total",
                               from_7array<unsigned>(npartTotal_));
        HDFWriteGroupAttribute(fname_, "Header", "NumPart_Total_HighWord",
                               from_7array<unsigned>(npartTotalHighWord_));
        HDFWriteGroupAttribute(fname_, "Header", "NumPart_ThisFile",
                               from_7array<uint64_t>(npart_));
        HDFWriteGroupAttribute(fname_, "Header", "MassTable",
                               from_7array<double>(mass_));

        HDFWriteGroupAttribute(fname_, "Header", "Time",
                               from_value<double>(time_));
        HDFWriteGroupAttribute(fname_, "Header", "Redshift",
                               from_value<double>(zstart_));
        HDFWriteGroupAttribute(fname_, "Header", "Flag_Entropy_ICs",
                               from_value<int>(0));

        HDFWriteGroupAttribute(fname_, "Header", "NumFilesPerSnapshot",
                               from_value<int>(num_files_));

        music::ilog << "Done writing SWIFT IC file to " << fname_ << std::endl;
        music::ilog << "Note that the IC file does not contain any h-factors "
                       "nor any extra sqrt(a)-factors for the velocities"
                    << std::endl;
        music::ilog
            << "The SWIFT parameters 'InitialConditions:cleanup_h_factors' and "
               "'InitialConditions:cleanup_velocity_factors' should hence "
               "*NOT* be set!"
            << std::endl;
      }
    }
  }

  /** @return output_type::particles for every supported species. */
  output_type write_species_as(const cosmo_species &) const {
    return output_type::particles;
  }

  /** @return Position conversion factor from box units to Mpc. */
  real_t position_unit() const { return lunit_; }

  /** @return Velocity conversion factor to km/s. */
  real_t velocity_unit() const { return vunit_; }

  /** @return Mass conversion factor to 1e10 solar masses. */
  real_t mass_unit() const { return munit_; }

  /** @return Whether particle floating-point fields are stored as doubles. */
  bool has_64bit_reals() const {
    if (typeid(write_real_t) == typeid(double))
      return true;
    return false;
  }

  /** @return Whether particle IDs are stored as unsigned 64-bit integers. */
  bool has_64bit_ids() const {
    if (blongids_)
      return true;
    return false;
  }

  /**
   * @brief Map a monofonIC species to its Gadget/SWIFT particle-type index.
   * @param s Species to map.
   * @return Particle-type index, or -1 for an unsupported species.
   */
  int get_species_idx(const cosmo_species &s) const {
    switch (s) {
    case cosmo_species::dm:
      return 1;
    case cosmo_species::baryon:
      return 0;
    case cosmo_species::neutrino:
      return 6;
    }
    return -1;
  }

  /**
   * @brief Sort and write one species, then emit its /Cells metadata.
   *
   * The method builds a stable local counting-sort permutation, combines cell
   * counts and bounds across ranks, derives each rank's global file offsets,
   * writes all particle fields with one shared ordering, and finally records
   * the species-specific cell index on rank zero.
   *
   * @param pc Local particle arrays plus local and global particle counts.
   * @param s Species represented by @p pc.
   * @param Omega_species Cosmological density parameter used for constant mass.
   */
  void write_particle_data(const particle::container &pc,
                           const cosmo_species &s, double Omega_species) {
    const int sid = get_species_idx(s);
    assert(sid != -1);
    const std::string grp = std::string("PartType") + std::to_string(sid);

    const size_t n_local = pc.get_local_num_particles();
    const size_t global_num_particles = pc.get_global_num_particles();

    npart_[sid] = global_num_particles;
    npartTotal_[sid] = (uint32_t)(global_num_particles);
    npartTotalHighWord_[sid] = (uint32_t)((global_num_particles) >> 32);

    const double particle_mass =
        pc.bhas_individual_masses_
            ? 0.0
            : Omega_species * munit_ / double(global_num_particles);

    const double t_start = get_wtime();

    //... sort the local particles into SWIFT's top-level cells ...........
    std::vector<size_t> perm;
    std::vector<swift_count_t> local_counts;
    std::vector<double> minpos, maxpos;

    if (this->has_64bit_reals())
      this->build_local_layout(pc.positions64_, n_local, perm, local_counts,
                               minpos, maxpos);
    else
      this->build_local_layout(pc.positions32_, n_local, perm, local_counts,
                               minpos, maxpos);

    //... turn that into the global, cell-major file layout ...............
    std::vector<swift_count_t> global_counts, cell_offset, rank_offset;
    this->global_cell_layout(local_counts, global_counts, cell_offset,
                             rank_offset, minpos, maxpos);

    //... the file ranges this rank contributes, adjacent cells merged ....
    std::vector<swift_write_range> ranges;
    for (size_t c = 0; c < ncells_; ++c) {
      if (local_counts[c] == 0)
        continue;
      const hsize_t fs = (hsize_t)rank_offset[c];
      if (!ranges.empty() &&
          ranges.back().file_start + ranges.back().length == fs)
        ranges.back().length += (hsize_t)local_counts[c];
      else
        ranges.push_back({fs, (hsize_t)local_counts[c]});
    }

    const double t_sorted = get_wtime();

    // rank 0 creates the full, empty datasets in the file
    if (this_rank_ == 0) {
      this->create_species_datasets(grp, global_num_particles, s);
      music::ilog << "Created empty arrays for " << grp << " into file "
                  << fname_ << "." << std::endl;
    }
    this->barrier();

    // every rank writes its own particles into their cells' slots
    this->write_species_fields(grp, pc, s, perm, ranges, particle_mass);
    this->barrier();

    const double t_written = get_wtime();

    // rank 0 records the cell metadata for this species
    if (this_rank_ == 0)
      this->write_species_cell_metadata(grp, global_counts, cell_offset, minpos,
                                        maxpos);
    this->barrier();

    this->log_io_timing(grp, s, global_num_particles, ranges.size(),
                        t_sorted - t_start, t_written - t_sorted,
                        get_wtime() - t_written);
  }

protected:
  /**
   * @brief Log maximum rank timings and estimated write throughput.
   * @param grp HDF5 particle group name.
   * @param s Species used to include optional gas fields in the byte estimate.
   * @param n_global Global particle count for the species.
   * @param n_ranges Number of disjoint file ranges written by this rank.
   * @param t_sort Local cell-layout construction time in seconds.
   * @param t_write Local particle-field write time in seconds.
   * @param t_meta Local metadata write and synchronisation time in seconds.
   *
   * MPI builds report maximum timings and range count across ranks. Rank zero
   * writes one machine-readable SWIFT-IO line per species.
   */
  void log_io_timing(const std::string &grp, const cosmo_species &s,
                     size_t n_global, size_t n_ranges, double t_sort,
                     double t_write, double t_meta) const {
    double tmax[3] = {t_sort, t_write, t_meta};
    unsigned long long ranges_max = n_ranges;
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, tmax, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ranges_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    if (this_rank_ != 0)
      return;

    const size_t nreal = this->has_64bit_reals() ? 8 : 4;
    const size_t nid = this->has_64bit_ids() ? 8 : 4;
    size_t per_particle = 6 * nreal + nid + nreal; // pos, vel, id, mass
    if (bdobaryons_ && s == cosmo_species::baryon)
      per_particle += 2 * sizeof(write_real_t);
    const double gbytes =
        double(n_global) * double(per_particle) / (1024. * 1024. * 1024.);

#ifdef USE_PARALLEL_HDF5
    const char *backend = "parallel";
#else
    const char *backend = "serial";
#endif

    music::ilog << "SWIFT-IO " << grp << " backend=" << backend
                << " nranks=" << num_ranks_ << " cdim=" << cdim_
                << " npart=" << n_global << " max_ranges=" << ranges_max
                << " t_sort=" << tmax[0] << " t_write=" << tmax[1]
                << " t_meta=" << tmax[2] << " GB=" << gbytes
                << " GB_per_s=" << (tmax[1] > 0. ? gbytes / tmax[1] : 0.)
                << std::endl;
  }

  /** @brief Synchronise all ranks; a no-op in non-MPI builds. */
  inline void barrier() const {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  /**
   * @brief Periodically wrap one coordinate into [0, box_out_).
   * @param x Coordinate in output length units.
   * @return Wrapped coordinate; a rounded upper boundary maps to zero.
   */
  inline double wrap_coord(double x) const {
    double y = std::fmod(x, box_out_);
    if (y < 0.0)
      y += box_out_;
    if (!(y < box_out_))
      y = 0.0; // rounding can land exactly on the upper edge
    return y;
  }

  /**
   * @brief Compute SWIFT's flattened top-level-cell index.
   * @param x Wrapped three-dimensional position in output length units.
   * @return Cell index with z varying fastest and x slowest.
   *
   * Coordinates are expected in [0, box_out_). The upper clamp protects
   * against floating-point roundoff at cell boundaries.
   */
  inline size_t cell_id(const double x[3]) const {
    size_t ijk[3];
    for (int d = 0; d < 3; ++d) {
      const size_t i = (size_t)(x[d] / cell_size_);
      ijk[d] = (i >= cdim_) ? cdim_ - 1 : i;
    }
    return (ijk[0] * cdim_ + ijk[1]) * cdim_ + ijk[2];
  }

  /**
   * @brief Build local cell counts, bounds, and a stable cell-order
   * permutation.
   * @tparam pos_t Floating-point type of the position array.
   * @param pos Flat local position array in (x, y, z) order.
   * @param n_local Number of local particles.
   * @param[out] perm Source indices ordered by cell, then original local order.
   * @param[out] counts Number of local particles in every cell.
   * @param[out] minpos Per-cell minima, flattened as (cell, dimension).
   * @param[out] maxpos Per-cell maxima, flattened as (cell, dimension).
   * @throws std::runtime_error If any coordinate is non-finite.
   *
   * Empty cells retain DBL_MAX and -DBL_MAX bounds, matching SWIFT's
   * sentinel convention. Positions are wrapped before assigning cells and
   * calculating bounds.
   */
  template <typename pos_t>
  void build_local_layout(const std::vector<pos_t> &pos, size_t n_local,
                          std::vector<size_t> &perm,
                          std::vector<swift_count_t> &counts,
                          std::vector<double> &minpos,
                          std::vector<double> &maxpos) const {
    counts.assign(ncells_, 0);
    minpos.assign(3 * ncells_, DBL_MAX);
    maxpos.assign(3 * ncells_, -DBL_MAX);

    std::vector<uint32_t> cellid(n_local);

    for (size_t i = 0; i < n_local; ++i) {
      double x[3];
      for (int d = 0; d < 3; ++d) {
        const double xd = (double)pos[3 * i + d];
        if (!std::isfinite(xd))
          throw std::runtime_error(
              "SWIFT output: encountered a non-finite particle coordinate.");
        x[d] = this->wrap_coord(xd);
      }

      const size_t c = this->cell_id(x);
      cellid[i] = (uint32_t)c;
      ++counts[c];

      for (int d = 0; d < 3; ++d) {
        if (x[d] < minpos[3 * c + d])
          minpos[3 * c + d] = x[d];
        if (x[d] > maxpos[3 * c + d])
          maxpos[3 * c + d] = x[d];
      }
    }

    // counting sort, stable so that the order inside a cell is the input order
    std::vector<size_t> next(ncells_);
    size_t acc = 0;
    for (size_t c = 0; c < ncells_; ++c) {
      next[c] = acc;
      acc += (size_t)counts[c];
    }

    perm.resize(n_local);
    for (size_t i = 0; i < n_local; ++i)
      perm[next[cellid[i]]++] = i;
  }

  /**
   * @brief Combine local cell layouts and derive global write offsets.
   * @param local_counts This rank's particle count in each cell.
   * @param[out] global_counts Global particle count in each cell.
   * @param[out] cell_offset Start of each cell in the species dataset.
   * @param[out] rank_offset Start of this rank's contribution to each cell.
   * @param[in,out] minpos Local bounds on input, global bounds on output.
   * @param[in,out] maxpos Local bounds on input, global bounds on output.
   *
   * rank_offset[c] equals the global start of cell c plus contributions
   * from lower-numbered ranks. This produces a deterministic cell-major,
   * rank-major layout without redistributing particles between ranks.
   */
  void global_cell_layout(const std::vector<swift_count_t> &local_counts,
                          std::vector<swift_count_t> &global_counts,
                          std::vector<swift_count_t> &cell_offset,
                          std::vector<swift_count_t> &rank_offset,
                          std::vector<double> &minpos,
                          std::vector<double> &maxpos) const {
    global_counts = local_counts;
    std::vector<swift_count_t> preceding(ncells_, 0);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, global_counts.data(), (int)ncells_,
                  MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(local_counts.data(), preceding.data(), (int)ncells_,
               MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (this_rank_ == 0)
      std::fill(preceding.begin(), preceding.end(),
                0); // MPI_Exscan leaves rank 0 undefined
    MPI_Allreduce(MPI_IN_PLACE, minpos.data(), (int)(3 * ncells_), MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, maxpos.data(), (int)(3 * ncells_), MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
#endif

    cell_offset.resize(ncells_);
    rank_offset.resize(ncells_);
    swift_count_t off = 0;
    for (size_t c = 0; c < ncells_; ++c) {
      cell_offset[c] = off;
      rank_offset[c] = off + preceding[c];
      off += global_counts[c];
    }
  }

  /**
   * @brief Reorder a flat scalar or vector field with a particle permutation.
   * @tparam T Field element type.
   * @param src Flat source array.
   * @param perm Source particle index for each output particle.
   * @param width Elements per particle, normally one or three.
   * @return Cell-ordered copy of @p src.
   */
  template <typename T>
  static std::vector<T> permute(const std::vector<T> &src,
                                const std::vector<size_t> &perm, size_t width) {
    std::vector<T> out(perm.size() * width);
    for (size_t i = 0; i < perm.size(); ++i)
      for (size_t d = 0; d < width; ++d)
        out[i * width + d] = src[perm[i] * width + d];
    return out;
  }

  /**
   * @brief Write one cell-ordered field into this rank's HDF5 hyperslabs.
   * @tparam T Dataset element type.
   * @param fid Open HDF5 file handle.
   * @param name Dataset path.
   * @param ranges Disjoint particle ranges owned by this rank.
   * @param data Packed field data corresponding to @p ranges in order.
   * @param width Elements per particle, normally one or three.
   * @param dxpl HDF5 transfer property list (H5P_DEFAULT or collective).
   * @throws std::runtime_error If the dataset cannot be opened or written.
   *
   * All ranges are ORed into one file-space selection. Ranks with no local
   * particles use null/empty selections but still participate in collective
   * writes, as required by parallel HDF5.
   */
  template <typename T>
  static void write_ranges(hid_t fid, const std::string &name,
                           const std::vector<swift_write_range> &ranges,
                           const std::vector<T> &data, size_t width,
                           hid_t dxpl) {
    hid_t dset = H5Dopen(fid, name.c_str());
    if (dset < 0)
      throw std::runtime_error("SWIFT output: cannot open dataset " + name);

    hid_t fspace = H5Dget_space(dset);
    hid_t mspace;

    if (ranges.empty()) {
      // ranks without local particles still take part in the (collective) write
      mspace = H5Screate(H5S_NULL);
      H5Sselect_none(fspace);
    } else {
      hsize_t mdims[2] = {(hsize_t)(data.size() / width), (hsize_t)width};
      mspace = H5Screate_simple(width > 1 ? 2 : 1, mdims, NULL);

      bool first = true;
      for (const auto &r : ranges) {
        hsize_t start[2] = {r.file_start, 0};
        hsize_t count[2] = {r.length, (hsize_t)width};
        H5Sselect_hyperslab(fspace, first ? H5S_SELECT_SET : H5S_SELECT_OR,
                            start, NULL, count, NULL);
        first = false;
      }
    }

    if (H5Dwrite(dset, GetDataType<T>(), mspace, fspace, dxpl,
                 data.empty() ? NULL : &data[0]) < 0)
      throw std::runtime_error("SWIFT output: failed to write dataset " + name);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(dset);
  }

  /**
   * @brief Attach SWIFT unit exponents to one particle dataset.
   * @param dset Dataset path.
   * @param uM Mass-unit exponent.
   * @param uL Length-unit exponent.
   * @param ut Time-unit exponent.
   * @param a_exp Cosmological scale-factor exponent.
   *
   * Current and temperature exponents, plus the Hubble-factor exponent, are
   * zero for every field written by this plugin.
   */
  void write_unit_attributes(const std::string &dset, double uM, double uL,
                             double ut, double a_exp) const {
    HDFWriteDatasetAttribute(fname_, dset, "U_M exponent",
                             std::vector<double>{uM});
    HDFWriteDatasetAttribute(fname_, dset, "U_L exponent",
                             std::vector<double>{uL});
    HDFWriteDatasetAttribute(fname_, dset, "U_t exponent",
                             std::vector<double>{ut});
    HDFWriteDatasetAttribute(fname_, dset, "U_I exponent",
                             std::vector<double>{0.0});
    HDFWriteDatasetAttribute(fname_, dset, "U_T exponent",
                             std::vector<double>{0.0});
    HDFWriteDatasetAttribute(fname_, dset, "a-scale exponent",
                             std::vector<double>{a_exp});
    HDFWriteDatasetAttribute(fname_, dset, "h-scale exponent",
                             std::vector<double>{0.0});
  }

  /**
   * @brief Create empty particle datasets and their unit metadata.
   * @param grp Particle group name, for example PartType1.
   * @param global_num_particles Dataset length.
   * @param s Species used to decide whether gas-only fields are needed.
   *
   * Called only by rank zero. Serial HDF5 output retains checksums and
   * compression. Parallel output disables filters because collective writes to
   * filtered datasets are not portable across supported HDF5 versions.
   */
  void create_species_datasets(const std::string &grp,
                               size_t global_num_particles,
                               const cosmo_species &s) const {
#ifdef USE_PARALLEL_HDF5
    // collective writes into filtered datasets are not portable across HDF5
    // builds
    const bool filter = false;
#else
    const bool filter = true;
#endif

    HDFCreateGroup(fname_, grp);

    if (this->has_64bit_reals()) {
      HDFCreateEmptyDatasetVector<double>(fname_, grp + "/Coordinates",
                                          global_num_particles, filter);
      HDFCreateEmptyDatasetVector<double>(fname_, grp + "/Velocities",
                                          global_num_particles, filter);
      HDFCreateEmptyDataset<double>(fname_, grp + "/Masses",
                                    global_num_particles, filter);
    } else {
      HDFCreateEmptyDatasetVector<float>(fname_, grp + "/Coordinates",
                                         global_num_particles, filter);
      HDFCreateEmptyDatasetVector<float>(fname_, grp + "/Velocities",
                                         global_num_particles, filter);
      HDFCreateEmptyDataset<float>(fname_, grp + "/Masses",
                                   global_num_particles, filter);
    }

    if (this->has_64bit_ids())
      HDFCreateEmptyDataset<uint64_t>(fname_, grp + "/ParticleIDs",
                                      global_num_particles, filter);
    else
      HDFCreateEmptyDataset<uint32_t>(fname_, grp + "/ParticleIDs",
                                      global_num_particles, filter);

    this->write_unit_attributes(grp + "/Coordinates", 0.0, 1.0, 0.0, 1.0);
    this->write_unit_attributes(grp + "/Velocities", 0.0, 1.0, -1.0, 0.0);
    this->write_unit_attributes(grp + "/Masses", 1.0, 0.0, 0.0, 0.0);
    this->write_unit_attributes(grp + "/ParticleIDs", 0.0, 0.0, 0.0, 0.0);

    if (bdobaryons_ && s == cosmo_species::baryon) {
      // note: despite this being a constant array we still need to handle it in
      // a distributed way
      HDFCreateEmptyDataset<write_real_t>(fname_, grp + "/InternalEnergy",
                                          global_num_particles, filter);
      HDFCreateEmptyDataset<write_real_t>(fname_, grp + "/SmoothingLength",
                                          global_num_particles, filter);
      this->write_unit_attributes(grp + "/InternalEnergy", 0.0, 2.0, -2.0, 0.0);
      this->write_unit_attributes(grp + "/SmoothingLength", 0.0, 1.0, 0.0, 1.0);
    }
  }

  /**
   * @brief Open the shared file and write all fields through the selected
   * backend.
   * @param grp Particle group name.
   * @param pc Source particle container.
   * @param s Particle species.
   * @param perm Stable local cell-order permutation.
   * @param ranges Global file ranges corresponding to the permuted particles.
   * @param particle_mass Constant particle mass, ignored for individual masses.
   *
   * Parallel-HDF5 builds perform one collective file session on all ranks.
   * Other builds serialise rank access and perform one file session per rank.
   */
  void write_species_fields(const std::string &grp,
                            const particle::container &pc,
                            const cosmo_species &s,
                            const std::vector<size_t> &perm,
                            const std::vector<swift_write_range> &ranges,
                            double particle_mass) const {
#ifdef USE_PARALLEL_HDF5
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t fid = H5Fopen(fname_.c_str(), H5F_ACC_RDWR, fapl);
    if (fid < 0)
      throw std::runtime_error("SWIFT output: cannot open " + fname_ +
                               " for collective writing.");

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);

    this->write_all_fields(fid, dxpl, grp, pc, s, perm, ranges, particle_mass);

    H5Pclose(dxpl);
    H5Fclose(fid);
    H5Pclose(fapl);
#else
    // no parallel HDF5: the ranks take turns on the shared file, each one
    // opening it once
    for (int rank = 0; rank < num_ranks_; ++rank) {
      this->barrier();
      if (rank != this_rank_)
        continue;

      hid_t fid = H5Fopen(fname_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
      if (fid < 0)
        throw std::runtime_error("SWIFT output: cannot open " + fname_ +
                                 " for writing.");
      this->write_all_fields(fid, H5P_DEFAULT, grp, pc, s, perm, ranges,
                             particle_mass);
      H5Fclose(fid);
    }
#endif
    if (this_rank_ == 0)
      music::ilog << "Wrote cell-ordered " << grp << " data to the IC file."
                  << std::endl;
  }

  /**
   * @brief Pack and write every dataset for one species using a shared
   * ordering.
   * @param fid Open HDF5 file handle.
   * @param dxpl Dataset transfer property list.
   * @param grp Particle group name.
   * @param pc Source particle container.
   * @param s Particle species.
   * @param perm Stable local cell-order permutation.
   * @param ranges Global file ranges owned by this rank.
   * @param particle_mass Constant mass used when no individual masses exist.
   *
   * Coordinates are periodically wrapped before writing so file values and
   * /Cells bounds describe exactly the same positions. Every other field is
   * permuted identically, preserving particle correspondence.
   */
  void write_all_fields(hid_t fid, hid_t dxpl, const std::string &grp,
                        const particle::container &pc, const cosmo_species &s,
                        const std::vector<size_t> &perm,
                        const std::vector<swift_write_range> &ranges,
                        double particle_mass) const {
    const size_t n_local = perm.size();

    //... positions, stored periodically wrapped so that they match the cell
    // metadata
    if (this->has_64bit_reals()) {
      auto buf = permute(pc.positions64_, perm, 3);
      for (auto &x : buf)
        x = this->wrap_coord(x);
      write_ranges(fid, grp + "/Coordinates", ranges, buf, 3, dxpl);
    } else {
      auto buf = permute(pc.positions32_, perm, 3);
      for (auto &x : buf)
        x = (float)this->wrap_coord((double)x);
      write_ranges(fid, grp + "/Coordinates", ranges, buf, 3, dxpl);
    }

    //... velocities
    if (this->has_64bit_reals())
      write_ranges(fid, grp + "/Velocities", ranges,
                   permute(pc.velocities64_, perm, 3), 3, dxpl);
    else
      write_ranges(fid, grp + "/Velocities", ranges,
                   permute(pc.velocities32_, perm, 3), 3, dxpl);

    //... ids
    if (this->has_64bit_ids())
      write_ranges(fid, grp + "/ParticleIDs", ranges,
                   permute(pc.ids64_, perm, 1), 1, dxpl);
    else
      write_ranges(fid, grp + "/ParticleIDs", ranges,
                   permute(pc.ids32_, perm, 1), 1, dxpl);

    //... masses
    if (pc.bhas_individual_masses_) {
      if (this->has_64bit_reals())
        write_ranges(fid, grp + "/Masses", ranges, permute(pc.mass64_, perm, 1),
                     1, dxpl);
      else
        write_ranges(fid, grp + "/Masses", ranges, permute(pc.mass32_, perm, 1),
                     1, dxpl);
    } else {
      if (this->has_64bit_reals())
        write_ranges(fid, grp + "/Masses", ranges,
                     std::vector<double>(n_local, particle_mass), 1, dxpl);
      else
        write_ranges(fid, grp + "/Masses", ranges,
                     std::vector<float>(n_local, (float)particle_mass), 1,
                     dxpl);
    }

    //... gas internal energy and smoothing length if baryons are enabled
    if (bdobaryons_ && s == cosmo_species::baryon) {
      write_ranges(fid, grp + "/InternalEnergy", ranges,
                   std::vector<write_real_t>(n_local, (write_real_t)ceint_), 1,
                   dxpl);
      write_ranges(fid, grp + "/SmoothingLength", ranges,
                   std::vector<write_real_t>(n_local, (write_real_t)h_), 1,
                   dxpl);
    }
  }

  /**
   * @brief Write species-independent /Cells groups, attributes, and centres.
   *
   * Called only by rank zero during construction. Cell centres use the same
   * flattened order as cell_id() and all per-species cell datasets.
   */
  void write_common_cell_metadata() const {
    HDFCreateGroup(fname_, "Cells");
    HDFCreateGroup(fname_, "Cells/Meta-data");
    HDFWriteGroupAttribute(fname_, "Cells/Meta-data", "nr_cells",
                           from_value<int>((int)ncells_));
    HDFWriteGroupAttribute(fname_, "Cells/Meta-data", "size",
                           std::vector<double>(3, cell_size_));
    HDFWriteGroupAttribute(fname_, "Cells/Meta-data", "dimension",
                           std::vector<int>(3, (int)cdim_));
    HDFWriteGroupAttribute(fname_, "Cells/Meta-data", "Origin",
                           std::vector<double>(3, 0.0));

    std::vector<double> centres(3 * ncells_);
    for (size_t ix = 0; ix < cdim_; ++ix)
      for (size_t iy = 0; iy < cdim_; ++iy)
        for (size_t iz = 0; iz < cdim_; ++iz) {
          const size_t c = (ix * cdim_ + iy) * cdim_ + iz;
          centres[3 * c + 0] = (double(ix) + 0.5) * cell_size_;
          centres[3 * c + 1] = (double(iy) + 0.5) * cell_size_;
          centres[3 * c + 2] = (double(iz) + 0.5) * cell_size_;
        }
    HDFWriteDatasetVector(fname_, "Cells/Centres", centres);

    HDFCreateGroup(fname_, "Cells/Files");
    HDFCreateGroup(fname_, "Cells/OffsetsInFile");
    HDFCreateGroup(fname_, "Cells/Counts");
    HDFCreateGroup(fname_, "Cells/MinPositions");
    HDFCreateGroup(fname_, "Cells/MaxPositions");
  }

  /**
   * @brief Write counts, offsets, file indices, and bounds for one species.
   * @param grp Particle group name used as each metadata dataset name.
   * @param global_counts Global particle count in each cell.
   * @param cell_offset Start of each cell in the species datasets.
   * @param minpos Global per-cell coordinate minima.
   * @param maxpos Global per-cell coordinate maxima.
   *
   * Called only by rank zero. Files is zero for every cell because this
   * plugin always writes one shared output file. Consumers must inspect
   * Counts before interpreting the sentinel bounds of empty cells.
   */
  void
  write_species_cell_metadata(const std::string &grp,
                              const std::vector<swift_count_t> &global_counts,
                              const std::vector<swift_count_t> &cell_offset,
                              const std::vector<double> &minpos,
                              const std::vector<double> &maxpos) const {
    // a single shared file, so every cell lives in file 0
    HDFWriteDataset(fname_, "Cells/Files/" + grp, std::vector<int>(ncells_, 0));
    HDFWriteDataset(fname_, "Cells/OffsetsInFile/" + grp, cell_offset);
    HDFWriteDataset(fname_, "Cells/Counts/" + grp, global_counts);
    // empty cells keep SWIFT's +/-DBL_MAX sentinels, consumers are to check
    // Counts first
    HDFWriteDatasetVector(fname_, "Cells/MinPositions/" + grp, minpos);
    HDFWriteDatasetVector(fname_, "Cells/MaxPositions/" + grp, maxpos);
  }
};

namespace {
output_plugin_creator_concrete<swift_output_plugin<double>> creator301("SWIFT");
} // namespace

#endif
