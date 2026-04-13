import numpy as np

from cacao_stomata_response.metrics import angle_degrees, cosine_similarity, gaussian_w2_distance


def test_cosine_similarity_for_identical_vectors():
    a = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cosine_similarity(a, a), 1.0)


def test_angle_for_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(angle_degrees(a, b), 90.0)


def test_gaussian_w2_distance_is_zero_for_identical_samples():
    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert np.isclose(gaussian_w2_distance(x, x), 0.0, atol=1e-4)
