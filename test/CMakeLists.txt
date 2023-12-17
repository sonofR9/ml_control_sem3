set(TEST_EXECUTABLE test_picewise_approximation)

add_executable(${TEST_EXECUTABLE} "")

target_include_directories(${TEST_EXECUTABLE} PRIVATE 
    "../src/"
)

target_link_libraries(${TEST_EXECUTABLE} PRIVATE 
    "GTest::gtest_main"
    "GTest::gmock_main"
)

target_sources(${TEST_EXECUTABLE}
    PRIVATE
    "test_piecewise_linear_approx.cpp"
)

include(GoogleTest)
gtest_discover_tests(${TEST_EXECUTABLE})
