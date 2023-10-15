from typing import Literal, Optional

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchtuples as tt

# for debugging
# from codetiming import Timer
# from IPython.core.debugger import set_trace


def std_from_t(ds: pd.DataFrame, t: float):
    """Rolling stdev closest from t.

    Args:
        ds (pd.DataFrame): Dataframe. Must contain "duration" and "std_surv" columns.
        t (float): Time.

    Returns:
        float: rolling std of `_t` such that `_t` is closest to `t` in `ds`.
    """
    idt = np.argmin(abs(t - ds["duration"]))
    return ds.iloc[idt]["std_surv"]


def sample_alive_from_dates(
    dates: pd.Series,
    at_risk_dict: dict,
    sample_mode: Literal["diff", "weighted", "percentage", "adadiff"],
    sample_value: float,
    durations_all: npt.ArrayLike,
    durations_survival: npt.ArrayLike,
    sd_per_time: Optional[pd.DataFrame] = None,
    n_control: int = 1,
):
    """Sample index from (eventually a subset of) living at time given in dates.

    Args:
        dates (pd.Series): Times
        at_risk_dict (dict): Array with index of alive in X matrix. Has a "time" entry.
        sample_mode (Literal["diff", "weighted", "percentage", "adadiff"]): Heuristic to subsample from the risk set.
        sample_value (float): Hyperparameter attached to `sample_mode`.
        durations_all (npt.ArrayLike): _description_
        sd_per_time (Optional[pd.DataFrame], optional): Dataframe of rolling standard deviation. Needed for "adadiff" only. Defaults to None.
        n_control (int, optional): Number of samples. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # # initialization
    lengths = np.array([at_risk_dict[x].shape[0] for x in dates])
    samp = np.empty((dates.size, n_control), dtype=int)
    samp.fill(np.nan)
    # heuristic
    # # real work
    # with Timer(name=f"{sample_mode}-heuristic", text="{name}: {milliseconds:.3f} ms"):
    if sample_mode == "percentage":
        idx = np.random.uniform(
            low=float(sample_value), size=(n_control, dates.size)
        ) * (lengths - 1)
        idx = idx.astype("int")  # just last percentage

        for it, time in enumerate(dates):
            samp[it, :] = at_risk_dict[time][
                idx[:, it]
            ]  # give index of randomely chosen element of the alives
    elif sample_mode in [
        "diff, " "adadiff"
    ]:  # same method, only std is adapted in "adadiff"
        if sample_mode == "adadiff":
            if sd_per_time is None:
                raise ValueError("Should provide `sd_pertime` in this case.")
            std_for_t_i = [
                std_from_t(sd_per_time, i) for i in dates
            ]  # rolling standard deviation corresponding to i
            get_thresh = lambda j: std_for_t_i[j]
        else:
            std = np.std(durations_survival)  # standard deviation of all train set
            get_thresh = lambda j: std
        for j, date in enumerate(dates):
            risks_d = at_risk_dict[date]
            durations_in_risk = durations_all[risks_d]
            thresh = float(sample_value * get_thresh[j])
            # for each index in risk set: 1 when outside survial space
            indices_with_1 = np.where(durations_in_risk >= date + thresh)
            if len(indices_with_1[0]) < n_control:
                idx = [len(risks_d) - 1] * n_control
            else:
                idx = np.random.choice(indices_with_1[0], n_control)
            samp[j] = risks_d[idx]
    elif sample_mode == "weighted":
        for j, date in enumerate(dates):
            risks_d = at_risk_dict[date]
            durations_in_risk = durations_all[risks_d]
            weights = [
                i / np.sum(durations_in_risk)
                for i in np.asarray(durations_in_risk).astype("float64")
            ]  # normalizing
            weights = np.asarray(weights).astype("float64")
            weights /= (
                weights.sum()
            )  # normalizing again (because random.choice is picky)
            idx = np.random.choice(a=lengths[j], size=n_control, p=weights)
            samp[j, :] = risks_d[idx]

    # with Timer(name="Baseline", text="{name}: {milliseconds:.3f} ms"):
    # calculate baseline j
    idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype("int")
    samp_baseline = np.empty((dates.size, n_control), dtype=int)
    samp_baseline.fill(np.nan)

    for it, time in enumerate(dates):
        samp_baseline[it, :] = at_risk_dict[time][idx[:, it]]

    return samp, samp_baseline


def make_at_risk_dict(durations):
    """Create dict(duration: indices) from sorted df.
    A dict mapping durations to indices.
    For each time => index of all individual alive.

    Arguments:
        durations {np.arrary} -- durations.
    """
    assert type(durations) is np.ndarray, "Need durations to be a numpy array"
    durations = pd.Series(durations)
    assert durations.is_monotonic_increasing, "Requires durations to be monotonic"
    allidx = durations.index.values
    keys = durations.drop_duplicates(keep="first")
    at_risk_dict = dict()
    for ix, t in keys.iteritems():
        at_risk_dict[t] = allidx[ix:]
    return at_risk_dict


class DurationSortedDataset(tt.data.DatasetTuple):
    """We assume the dataset contrain `(input, durations, events)`, and
    sort the batch based on descending `durations`.

    See `torchtuples.data.DatasetTuple`.
    """

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        input, (duration, event) = batch
        idx_sort = duration.sort(descending=True)[1]
        event = event.float()
        batch = tt.tuplefy(input, event).iloc[idx_sort]
        return batch


class CoxCCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input,
        durations: npt.ArrayLike,
        events: npt.ArrayLike,
        sample_mode: Literal["diff", "weighted", "percentage", "adadiff"],
        sample_value: float,
        sd_per_time: Optional[pd.DataFrame] = None,
        n_control: int = 1,
    ):
        df_train_target = pd.DataFrame(dict(duration=durations, event=events))
        self.durations = df_train_target.loc[lambda x: x["event"] == 1]["duration"]
        self.at_risk_dict = make_at_risk_dict(durations)
        self.sample_mode = sample_mode
        self.sample_value = sample_value
        self.sd_per_time = sd_per_time
        self.durations_all = durations

        self.input = tt.tuplefy(input)
        assert type(self.durations) is pd.Series
        self.n_control = n_control

    def __getitem__(self, index):
        if (not hasattr(index, "__iter__")) and (type(index) is not slice):
            index = [index]
        fails = self.durations.iloc[index]
        x_case = self.input.iloc[fails.index]
        control_idx, control_idx_baseline = sample_alive_from_dates(
            fails.values,
            self.at_risk_dict,
            self.sample_mode,
            self.sample_value,
            self.durations_all,
            self.durations,
            self.sd_per_time,
            self.n_control,
        )

        x_control = tt.TupleTree(
            self.input.iloc[idx] for idx in control_idx.transpose()
        )
        x_control_baseline = tt.TupleTree(
            self.input.iloc[idx] for idx in control_idx_baseline.transpose()
        )
        return (
            tt.tuplefy(x_case, x_control).to_tensor(),
            tt.tuplefy(x_case, x_control_baseline).to_tensor(),
        )

    def __len__(self):
        return len(self.durations)


class CoxTimeDataset(CoxCCDataset):
    def __init__(
        self,
        input,
        durations: npt.ArrayLike,
        events: npt.ArrayLike,
        sample_mode: Literal["diff", "weighted", "percentage", "adadiff"],
        sample_value: float,
        sd_per_time: pd.DataFrame = None,
        n_control: int = 1,
    ):
        super().__init__(
            input, durations, events, sample_mode, sample_value, sd_per_time, n_control
        )
        self.durations_tensor = tt.tuplefy(
            self.durations.values.reshape(-1, 1)
        ).to_tensor()

    def __getitem__(self, index):
        if not hasattr(index, "__iter__"):
            index = [index]
        durations = self.durations_tensor.iloc[index]

        (case_train, control_train), (case_val, control_val) = super().__getitem__(
            index
        )
        case_train = case_train + durations
        control_train = control_train.apply_nrec(lambda x: x + durations)
        case_val = case_val + durations
        control_val = control_val.apply_nrec(lambda x: x + durations)
        return tt.tuplefy(case_train, control_train), tt.tuplefy(case_val, control_val)


@numba.njit
def _pair_rank_mat(mat, idx_durations, events, dtype="float32"):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat


def pair_rank_mat(idx_durations, events, dtype="float32"):
    """Indicator matrix R with R_ij = 1{T_i < T_j and D_i = 1}.
    So it takes value 1 if we observe that i has an event before j and zero otherwise.

    Arguments:
        idx_durations {np.array} -- Array with durations.
        events {np.array} -- Array with event indicators.

    Keyword Arguments:
        dtype {str} -- dtype of array (default: {'float32'})

    Returns:
        np.array -- n x n matrix indicating if i has an observerd event before j.
    """
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat


class DeepHitDataset(tt.data.DatasetTuple):
    def __getitem__(self, index):
        input, target = super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)
