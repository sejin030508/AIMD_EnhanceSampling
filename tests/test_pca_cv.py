import numpy as np

from confmh.pca_cv import PCACV


def test_projection_is_rigid_transform_invariant():
    reference = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 1.0]]
    )
    mean = reference.reshape(-1)
    component = np.linspace(-1.0, 1.0, reference.size)
    component /= np.linalg.norm(component)
    cv = PCACV(
        reference_ca_nm=reference,
        mean_flat=mean,
        components=component[None, :],
        score_mean=np.zeros(1),
        score_scale=np.ones(1),
    )
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transformed = reference @ rotation + np.array([4.0, -3.0, 2.0])
    assert np.allclose(cv.project_ca(reference), cv.project_ca(transformed), atol=1e-8)
