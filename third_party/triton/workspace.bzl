"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "ae384d777d47baa798c77e0cec706f5cf6c5de54"
    TRITON_SHA256 = "f2f120d49c4acf152f9c1c1e88f25e7d4b30cc075d80bb9433c3859c3a51d875"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = ["//third_party/triton:cl518645628.patch"],
    )
