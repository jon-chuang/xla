// RUN: mlir-hlo-opt %s --split-input-file \
// RUN:   --gml-st-cpu-tiling-pipeline=matmul-tile-sizes=4,5,6 \
// RUN:   --canonicalize | FileCheck %s

func.func @matvec(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @matvec
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[C17:.*]] = arith.constant 17 : index
// CHECK-DAG:     %[[C32:.*]] = arith.constant 32 : index
// CHECK:         gml_st.parallel {{.*}} (%[[C0]]) to (%[[C32]]) step (%[[C4]])
// CHECK:           scf.for {{.*}} %[[C0]] to %[[C12]] step %[[C6]]
// CHECK:             vector.contract {{.*}} vector<4x6xf32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<4xf32>
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C17]] step %[[C6]]
// CHECK:           linalg.matvec

// -----

func.func @vecmat(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                  %output: tensor<33xf32>) -> tensor<33xf32> {
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%output : tensor<33xf32>) -> tensor<33xf32>
  return %2 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C12:.*]] = arith.constant 12 : index
// CHECK-DAG:     %[[C17:.*]] = arith.constant 17 : index
// CHECK-DAG:     %[[C30:.*]] = arith.constant 30 : index
// CHECK:         gml_st.parallel {{.*}} (%[[C0]]) to (%[[C30]]) step (%[[C5]])
// CHECK:           scf.for {{.*}} %[[C0]] to %[[C12]] step %[[C6]]
// CHECK:             vector.contract {{.*}} vector<6x5xf32>
// CHECK-NEXT:        scf.yield %{{.*}} : vector<5xf32>
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C17]] step %[[C6]]
// CHECK:           linalg.vecmat

// -----

func.func @dot(%lhs: tensor<19xf32>, %rhs: tensor<19xf32>,
               %output: tensor<f32>) -> tensor<f32> {
  %2 = linalg.dot ins(%lhs, %rhs : tensor<19xf32>, tensor<19xf32>)
                  outs(%output : tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: @dot
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C18:.*]] = arith.constant 18 : index
// CHECK:         scf.for {{.*}} %[[C0]] to %[[C18]] step %[[C6]]
// CHECK:           vector.contract {{.*}} vector<6xf32>
// CHECK-NEXT:      vector.broadcast
// CHECK-NEXT:      scf.yield %{{.*}} : vector<f32>
// CHECK:         arith.mulf
// CHECK:         arith.addf

// -----

func.func @matvec_addf(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>,
                       %add: tensor<33xf32>) -> tensor<33xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<33xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%1 : tensor<33xf32>) -> tensor<33xf32>
  %3 = linalg.map { arith.addf } ins(%2, %add : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  %4 = linalg.map { arith.addf } ins(%3, %add : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  return %4 : tensor<33xf32>
}

// CHECK-LABEL: @matvec_addf
// CHECK-SAME:  (%{{.*}}: {{.*}}, %{{.*}}: {{.*}}, %[[ARG:.*]]: tensor<33xf32>)
// CHECK:         gml_st.parallel {{.*}} outs (%[[ARG_PAR:.*]] = %[[ARG]]
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG_PAR]]
// CHECK:           %[[READ_INIT:.*]] = vector.transfer_read %[[ARG_PAR]]
// CHECK:           %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG_FOR:.*]] = %[[READ_INIT]]
// CHECK:             vector.contract {{.*}} %[[ARG_FOR]] :
// CHECK-NEXT:        scf.yield
// CHECK:           %[[CONTRACT:.*]] = vector.contract {{.*}} %[[FOR]] :
// CHECK:           vector.transfer_write %[[CONTRACT]], %[[EXTRACT]]
// CHECK:           gml_st.set_yield
// CHECK:         scf.for
// CHECK:           linalg.matvec
// CHECK:         gml_st.parallel
// CHECK:           arith.addf
// CHECK-NOT:       arith.addf
// CHECK:           gml_st.set_yield
// CHECK:         arith.addf
// CHECK-NOT:     arith.addf

// -----

func.func @matvec_no_dominate_addf(%lhs: tensor<33x17xf32>, %rhs: tensor<17xf32>) -> tensor<33xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 1.000000e+00 : f32
  %0 = tensor.empty() : tensor<33xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
  %2 = linalg.matvec ins(%lhs, %rhs : tensor<33x17xf32>, tensor<17xf32>)
                     outs(%1 : tensor<33xf32>) -> tensor<33xf32>
  %3 = tensor.empty() : tensor<33xf32>
  %4 = linalg.fill ins(%cst1 : f32) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
  %5 = linalg.map { arith.addf } ins(%2, %4 : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  return %5 : tensor<33xf32>
}

// CHECK-LABEL: @matvec_no_dominate_addf
// CHECK:         gml_st.parallel
// CHECK:           scf.for
// CHECK:             vector.contract
// CHECK-NEXT:        scf.yield
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield
// CHECK:         scf.for
// CHECK:           linalg.matvec
// CHECK:         gml_st.parallel
// CHECK:           arith.addf
// CHECK:           gml_st.set_yield
// CHECK:         arith.addf

// -----

func.func @vecmat_addf(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                       %add: tensor<33xf32>) -> tensor<33xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<33xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%1 : tensor<33xf32>) -> tensor<33xf32>
  %3 = linalg.map { arith.addf } ins(%add, %2 : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  %4 = linalg.map { arith.addf } ins(%3, %add : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  return %4 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat_addf
// CHECK-SAME:  (%{{.*}}: {{.*}}, %{{.*}}: {{.*}}, %[[ARG:.*]]: tensor<33xf32>)
// CHECK:         gml_st.parallel {{.*}} outs (%[[ARG_PAR:.*]] = %[[ARG]]
// CHECK:           %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG_PAR]]
// CHECK:           %[[READ_INIT:.*]] = vector.transfer_read %[[ARG_PAR]]
// CHECK:           %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG_FOR:.*]] = %[[READ_INIT]]
// CHECK:             vector.contract {{.*}} %[[ARG_FOR]] :
// CHECK-NEXT:        scf.yield
// CHECK:           %[[CONTRACT:.*]] = vector.contract {{.*}} %[[FOR]] :
// CHECK:           vector.transfer_write %[[CONTRACT]], %[[EXTRACT]]
// CHECK:           gml_st.set_yield
// CHECK:         scf.for
// CHECK:           linalg.vecmat
// CHECK:         gml_st.parallel
// CHECK:           arith.addf
// CHECK-NOT:       arith.addf
// CHECK:           gml_st.set_yield
// CHECK:         arith.addf
// CHECK-NOT:     arith.addf

// -----

func.func @vecmat_multiple_uses_addf(%lhs: tensor<17xf32>, %rhs: tensor<17x33xf32>,
                                     %add: tensor<33xf32>) -> tensor<33xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<33xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<33xf32>) -> tensor<33xf32>
  %2 = linalg.vecmat ins(%lhs, %rhs : tensor<17xf32>, tensor<17x33xf32>)
                     outs(%1 : tensor<33xf32>) -> tensor<33xf32>
  %3 = linalg.map { arith.addf } ins(%add, %2 : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  %4 = linalg.map { arith.addf } ins(%2, %3 : tensor<33xf32>, tensor<33xf32>) outs(%0 : tensor<33xf32>)
  return %4 : tensor<33xf32>
}

// CHECK-LABEL: @vecmat_multiple_uses_addf
// CHECK:         gml_st.parallel
// CHECK:           scf.for
// CHECK:             vector.contract
// CHECK-NEXT:        scf.yield
// CHECK:           vector.contract
// CHECK:           vector.transfer_write
// CHECK:           gml_st.set_yield
// CHECK:         scf.for
// CHECK:           linalg.vecmat
// CHECK:         gml_st.parallel
// CHECK:           arith.addf
// CHECK:           arith.addf
// CHECK:           gml_st.set_yield
// CHECK:         arith.addf
// CHECK:         arith.addf

// -----

func.func @dot_addf(%lhs: tensor<19xf32>, %rhs: tensor<19xf32>,
                    %arg: tensor<19xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<f32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
  %2 = linalg.dot ins(%lhs, %rhs : tensor<19xf32>, tensor<19xf32>)
                  outs(%1 : tensor<f32>) -> tensor<f32>
  %3 = linalg.dot ins(%arg, %lhs : tensor<19xf32>, tensor<19xf32>)
                  outs(%1 : tensor<f32>) -> tensor<f32>
  %4 = linalg.map { arith.addf } ins(%3, %2 : tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>)
  return %4 : tensor<f32>
}

// CHECK-LABEL: @dot_addf
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:     %[[C18:.*]] = arith.constant 18 : index
// CHECK:         scf.for
// CHECK:           vector.contract
// CHECK-NEXT:      vector.broadcast
// CHECK-NEXT:      scf.yield
// CHECK:         arith.mulf
// CHECK:         %[[RES_ADD:.*]] = arith.addf
// CHECK:         %[[RES_DOT:.*]] = tensor.from_elements %[[RES_ADD]]
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[RES_DOT]]
// CHECK:         %[[FOR:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[READ]]
// CHECK:           vector.contract
// CHECK-NEXT:      vector.broadcast
// CHECK-NEXT:      scf.yield
// CHECK:         vector.transfer_write %[[FOR]], %[[RES_DOT]]
// CHECK:         arith.mulf
// CHECK:         arith.addf
// CHECK-NOT:     arith.addf
