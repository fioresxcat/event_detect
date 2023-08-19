import torchmetrics
import torch
import pdb


class NormalAccuracy(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        max_preds, max_pred_indices = torch.max(preds, dim=1)
        valid_pred_indices = max_pred_indices[max_preds>=0.5]
        max_target, max_target_indices = torch.max(target, dim=1)
        valid_target_indices = max_target_indices[max_preds>=0.5]

        # n_true = (valid_pred_indices==valid_target_indices).sum()
        n_true = (max_pred_indices==max_target_indices).sum()

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total
    

    def get_false_indices(self, preds: torch.Tensor, target: torch.Tensor, offset: int=4):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        max_preds, max_pred_indices = torch.max(preds, dim=1)
        valid_pred_indices = max_pred_indices[max_preds>=0.5]
        max_target, max_target_indices = torch.max(target, dim=1)
        valid_target_indices = max_target_indices[max_preds>=0.5]

        false_indices = (max_pred_indices!=max_target_indices).nonzero(as_tuple=True)[0]
        pred_probs = preds[false_indices]
        gt_probs = target[false_indices]
        
        # extended_false_indices = []
        # for idx in false_indices:
        #     extended_false_indices.extend(list(range(idx-offset, idx+offset+1)))
        # extended_false_indices = torch.tensor(extended_false_indices, dtype=torch.long)
        # extended_pred_probs = preds[extended_false_indices]
        # extended_gt_probs = target[extended_false_indices]

        # return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist(), extended_false_indices, extended_pred_probs.cpu().numpy().tolist(), extended_gt_probs.cpu().numpy().tolist()
        return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist()




class MyAccuracy_2(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self, ev_diff_thresh):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        n_true = 0
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)
            if max_pred_idx == max_true_idx:
                n_true += 1
            elif max_true_value < 0.5 and max_true_idx != 2:
                if max_pred_idx == 2:
                    n_true += 1
            

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class MyAccuracy_3(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self, ev_diff_thresh):
        super().__init__()
        self.ev_diff_thresh = ev_diff_thresh
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds = torch.softmax(preds, dim=1)
        preds = torch.sigmoid(preds)

        # filter only rows that have 1 in its values
        preds = preds[(target==1).any(dim=1)]
        target = target[(target==1).any(dim=1)]
        print(preds)
        print(target)
        max_pred_indices = torch.argmax(preds, dim=1)
        max_true_indices = torch.argmax(target, dim=1)
        n_true = (max_pred_indices==max_true_indices).sum()

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total


class RelaxedAccuracy(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        n_true = 0
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)
            if max_pred_idx == max_true_idx:
                n_true += 1
                continue
            
            if (true[-1] != 1):
                true_ev_idx = torch.where(true[:2] > 0)[0][0]
                if max_pred_idx == true_ev_idx:
                    n_true += 1
                    continue
            

        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total
    

    def get_false_indices(self, preds: torch.Tensor, target: torch.Tensor, offset: int=4):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        false_indices = []
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)
            if max_pred_idx == max_true_idx:
                continue
            
            if (true[-1] != 1):
                true_ev_idx = torch.where(true[:2] > 0)[0][0]
                if max_pred_idx == true_ev_idx:
                    continue
            
            false_indices.append(i)
        
        pred_probs = preds[false_indices]
        gt_probs = target[false_indices]

        return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist()


class MoreRelaxedAccuracy(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        n_true = 0
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)



        self.correct += n_true
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total
    

    def get_false_indices(self, preds: torch.Tensor, target: torch.Tensor, offset: int=4):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        false_indices = []
        for i in range(len(preds)):
            pred = preds[i]
            true = target[i]
            max_pred_value, max_pred_idx = torch.max(pred, dim=0)
            max_true_value, max_true_idx = torch.max(true, dim=0)
            if max_pred_idx == max_true_idx:
                continue
            
            if (true[-1] != 1):
                true_ev_idx = torch.where(true[:2] > 0)[0][0]
                if max_pred_idx == true_ev_idx:
                    continue
            
            false_indices.append(i)
        
        pred_probs = preds[false_indices]
        gt_probs = target[false_indices]

        return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist()


class PCE(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True


    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        preds = preds[:, :2]
        target = target[:, :2]

        # threshold 0.5
        preds = (preds > self.threshold).float()
        target = (target > self.threshold).float()

        n_true = torch.sum(preds == target)

        self.correct += n_true
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
    

    def get_false_indices(self, preds: torch.Tensor, target: torch.Tensor, offset: int=4):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        preds_thresholded = (preds > self.threshold).float()
        target_thresholded = (target > self.threshold).float()


        # get max indices
        max_pred_indices = torch.argmax(preds_thresholded, dim=1)
        max_target_indices = torch.argmax(target_thresholded, dim=1)

        false_indices = torch.where(max_pred_indices != max_target_indices)[0]
        
        pred_probs = preds[false_indices]
        gt_probs = target[false_indices]

        return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist()


class SmoothPCE(torchmetrics.Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update = True


    def __init__(self, ev_diff_thresh):
        super().__init__()
        self.ev_diff_thresh = ev_diff_thresh
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        # only consider the first 2 elements
        preds = preds[:, :2]
        target = target[:, :2]
        diff = preds - target
        n_true = torch.sum(torch.abs(diff) < self.ev_diff_thresh)

        self.correct += n_true
        self.total += target.numel()


    def compute(self):
        return self.correct.float() / self.total
    

    def get_false_indices(self, preds: torch.Tensor, target: torch.Tensor, offset: int=4):
        preds = torch.softmax(preds, dim=1)
        # preds = torch.sigmoid(preds)

        diff = preds - target
        false_indices = torch.where((torch.abs(diff) > self.ev_diff_thresh).any(dim=1))[0]
        
        pred_probs = preds[false_indices]
        gt_probs = target[false_indices]

        return false_indices, pred_probs.cpu().numpy().tolist(), gt_probs.cpu().numpy().tolist()



