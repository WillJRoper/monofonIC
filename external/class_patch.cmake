# Patch the fetched CLASS sources. Run as a FetchContent PATCH_COMMAND with
#   cmake -DCLASS_SOURCE_DIR=<SOURCE_DIR> -P class_patch.cmake
#
# CLASS fills its log(a) background table as
#   loga_ini + i*(loga_final-loga_ini)/(bt_size-1)
# so rounding can leave the last entry a tiny distance from log(a/a_0) = 0
# (seen with the Intel compilers' default floating-point model). z_table then
# ends at ~1e-15 rather than 0, and CLASS rejects background lookups at z = 0
# in background_tau_of_z and background_at_z. The patch sets the last entry to
# exactly loga_final. It is idempotent, so re-running the patch step is safe.
#
# This is a local workaround for running CANVAS, not a fix intended for
# upstream monofonIC; the proper fix belongs in CLASS itself.

set(file "${CLASS_SOURCE_DIR}/source/background.c")
file(READ "${file}" text)

set(old [=[pba->loga_table[index_loga] = loga_ini + index_loga*(loga_final-loga_ini)/(pba->bt_size-1);]=])
set(new [=[pba->loga_table[index_loga] = (index_loga == pba->bt_size-1) ? loga_final : loga_ini + index_loga*(loga_final-loga_ini)/(pba->bt_size-1);]=])

string(FIND "${text}" "${old}" pos)
if(pos EQUAL -1)
  # Already patched, or CLASS has changed and the patch needs revisiting
  string(FIND "${text}" "${new}" patched)
  if(patched EQUAL -1)
    message(FATAL_ERROR "class_patch.cmake: log(a) table assignment not found in ${file}")
  endif()
  return()
endif()

string(REPLACE "${old}" "${new}" text "${text}")
file(WRITE "${file}" "${text}")
message(STATUS "Patched CLASS background.c: log(a) table ends exactly at a = a_0")
