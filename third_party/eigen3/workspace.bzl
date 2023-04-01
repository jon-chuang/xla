"""Provides the repository macro to import Eigen."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Eigen."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    EIGEN_COMMIT = "766db0202095527d2da0c9eecb829e345dc39279"
    EIGEN_SHA256 = "fb31a7a71755bb3621371c9dd5996ad1069eda7c0faf4316eadebc19da84d911"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/eigen.cmake)

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT)),
    )
