







mmpose/udp: [mmpose/mmpose/datasets/pipelines/top_down_transform.py](https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/pipelines/top_down_transform.py)



```python
def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).
        Note:
            num keypoints: K
            heatmap height: H
            heatmap width: W
            num target channels: C
            C = K if target_type=='GaussianHeatMap'
            C = 3*K if target_type=='CombinedTarget'
        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatMap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatMap' or 'CombinedTarget'.
                GaussianHeatMap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
        Returns:
            tuple: A tuple containing targets.
            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        assert target_type in ['GaussianHeatMap', 'CombinedTarget']

        if target_type == 'GaussianHeatMap':
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        elif target_type == 'CombinedTarget':
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            ##feat_x_int, feat_y_int的维度都变成[feat_height, feat_width]
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            ##feat_x_int, feat_y_int的维度都变成[feat_height*feat_width,]
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]  ##对应论文中的m
                mu_y = joints_3d[joint_id][1] / feat_stride[1]  ##对应论文中的n
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1  ##对应论文中的C
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]  ##对应论文中的X
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]  ##对应论文中的Y
            target = target.reshape(num_joints * 3, heatmap_size[1], heatmap_size[0])

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight
```

