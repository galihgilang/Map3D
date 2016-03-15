import numpy as np

def compute_best_rotation(corrs_from, corrs_to, points_from=None, allow_mirror=True):
    # includes mirroring !

    assert corrs_from.shape == corrs_to.shape


    # rotation
    H = corrs_to.T.dot(corrs_from).T

    U, S, Vt = np.linalg.svd(H)

    rot = Vt.T.dot(U.T)

    if not allow_mirror and np.linalg.det(rot) < -0.5:
        assert np.argmin(S) == S.size-1

        Vt[-1, :] = -Vt[-1, :]

        rot = Vt.T.dot(U.T)

        assert np.linalg.det(rot) >= -0.5



    if points_from is not None:
        points_from_deformed = rot.dot(points_from.T).T

        return rot, points_from_deformed
    return rot




def compute_best_rigid_deformation(corrs_from, corrs_to, points_from, points_to):
    from_mean, to_mean = points_from.mean(axis=0), points_to.mean(axis=0)

    corrs_from_n = corrs_from - from_mean[None, :]
    corrs_to_n = corrs_to - to_mean[None, :]

    rot = compute_best_rotation(corrs_from_n, corrs_to_n)

    points_from_deformed = rot.dot((points_from - from_mean[None, :]).T).T + to_mean[None, :]

    return from_mean, to_mean, rot, points_from_deformed

