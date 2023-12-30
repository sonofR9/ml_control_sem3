function(find_boost_libs LIBRARIES)
#   # Find Boost using find_package
#   find_package(Boost 1.84 CONFIG COMPONENTS ${LIBRARIES} QUIET)

#   # check if each library was found
#   set(FOUND_ALL TRUE)
#   if (${Boost_FOUND})
#     foreach(LIBRARY ${LIBRARIES})
#         if(NOT ${Boost_${LIBRARY}_FOUND})
#             set(FOUND_ALL FALSE)
#             message("Boost ${LIBRARY} was not found")
#         endif()
#     endforeach()
#   else()
#     set(FOUND_ALL FALSE)
#     message(Boost was not found)
#   endif()

#   # If Boost was found, return
#   if(${FOUND_ALL})
#     return()
#   elseif(${Boost_FOUND})
#     foreach(LIBRARY ${LIBRARIES})
#         if (NOT ${Boost_${LIBRARY}_FOUND})
#             continue()
#         endif()

#         set_target_properties(Boost::${LIBRARY} PROPERTIES OUTPUT_NAME rm_Boost_${LIBRARY})
#         unset(Boost::${LIBRARY})
#         unset(Boost_${LIBRARY}_FOUND)
#         unset(Boost_${LIBRARY}_LIBRARY)
#         unset(Boost_${LIBRARY}_INCLUDE_DIR)
#         unset(Boost_${LIBRARY})
#     endforeach()
#   endif()

  # Otherwise, fetch Boost using FetchContent
  set(BOOST_INCLUDE_LIBRARIES ${LIBRARIES})
  set(BOOST_ENABLE_CMAKE ON)

  include(FetchContent)
  FetchContent_Declare(Boost
    URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.7z
    USES_TERMINAL_DOWNLOAD TRUE
    GIT_PROGRESS TRUE
    DOWNLOAD_NO_EXTRACT FALSE
  )
  FetchContent_MakeAvailable(Boost)

  # Ensure Boost_INCLUDE_DIRS is set correctly
  if(Boost_INCLUDE_DIRS STREQUAL "Boost_INCLUDE_DIR-NOTFOUND")
    set(Boost_INCLUDE_DIRS "${PROJECT_BINARY_DIR}/_deps/boost-build/libs")
  endif()
endfunction()
