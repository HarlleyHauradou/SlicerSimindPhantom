cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerSimindPhantom)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/HarlleyHauradou/SlicerSimindPhantom")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "SlicerSimindPhantom contributors")
set(EXTENSION_DESCRIPTION "Patient-specific voxel phantoms for the SIMIND Monte Carlo code: CT -> TotalSegmentator -> Zubal-type phantom (.dat/.zub) or density (.dmi) + activity (.smi) maps.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/HarlleyHauradou/SlicerSimindPhantom/main/SimindPhantom/Resources/Icons/SimindPhantom.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS TotalSegmentator)

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(SimindPhantom)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
