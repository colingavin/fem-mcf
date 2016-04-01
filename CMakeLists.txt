CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

INCLUDE_DIRECTORIES(/usr/local/opt/nlopt/include)
LINK_DIRECTORIES(/usr/local/opt/nlopt/lib)

FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(RelaxationMCF)

ADD_EXECUTABLE(CircleTestProblem src/CSFSolver.cpp test/CircleTestProblem.cpp)
DEAL_II_SETUP_TARGET(CircleTestProblem)

ADD_EXECUTABLE(PaperclipTestProblem src/CSFSolver.cpp test/PaperclipTestProblem.cpp)
DEAL_II_SETUP_TARGET(PaperclipTestProblem)
TARGET_LINK_LIBRARIES(PaperclipTestProblem nlopt)

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )
