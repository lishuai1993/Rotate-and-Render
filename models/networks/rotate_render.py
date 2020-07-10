# from models.networks.render import *
from .render import *
import numpy as np
import cv2

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


class TestRender(Render):

    def __init__(self, opt):
        super(TestRender, self).__init__(opt)
        self.keypoints_106 = _load(osp.join(self.d, '106_index.npy'))

    def torch_get_106_points(self, vertices):
        vertixes = vertices.transpose(1, 2).contiguous()
        vertixes = vertixes.view(vertixes.size(0), -1)
        vertice_106 = vertixes[:, self.keypoints_106].reshape(vertices.size(0), -1, 3)
        return vertice_106

    def rotate_render(self, params, images, M=None, with_BG=False, pose_noise=False, large_pose=False, 
                      align=True, frontal=True, erode=True, grey_background=False, avg_BG=True,
                      yaw_pose=None, pitch_pose=None):

        bz, c, w, h = images.size()


        face_size = self.faces.size()
        self.faces_use = self.faces.expand(bz, face_size[1], face_size[2])

        # get render color vertices and normal vertices information, get original texs
        vertices = []
        vertices_out = []
        vertices_in_ori_img = []
        vertices_aligned_normal = []
        vertices_aligned_out = []
        vertices_ori_normal = []
        texs = []
        imgtemp = 0
        original_angles = torch.zeros(bz)
        with torch.no_grad():
            for n in range(bz):
                tex_a, vertice, vertice_out, vertice_in_ori_img, align_vertice, original_angle \
                    = self._forward(params[n], images[n], M[n],
                                    pose_noise=pose_noise, align=align, frontal=frontal, 
                                    yaw_pose=yaw_pose, pitch_pose=pitch_pose)                       # 完成了3d fitting、Rotate，
                vertices.append(vertice)
                vertices_out.append(vertice_out)
                vertices_in_ori_img.append(vertice_in_ori_img.clone())
                vertice2 = self.flip_normalize_vertices(vertice_in_ori_img.clone())
                vertices_ori_normal.append(vertice2)
                vertices_aligned_out.append(align_vertice)
                align_vertice_normal = self.flip_normalize_vertices(align_vertice.clone())
                vertices_aligned_normal.append(align_vertice_normal.clone())
                texs.append(tex_a)
                original_angles[n] = original_angle

            vertices = torch.cat(vertices, 0)
            vertices_aligned_normal = torch.cat(vertices_aligned_normal, 0)
            vertices_ori_normal = torch.cat(vertices_ori_normal, 0)

            vertices_in_ori_img = torch.stack(vertices_in_ori_img, 0)
            vertices_aligned_out = torch.stack(vertices_aligned_out, 0)

            texs = torch.cat(texs, 0)

            # erode the original mask and render again
            rendered_images_erode = None
            # if erode:
            #     with torch.cuda.device(self.current_gpu):               # 这里的vertices_ori_normal是已经按照目标姿态旋转过的顶点模型 ？
            #         rendered_images, depths, masks, = self.renderer(vertices_ori_normal, self.faces_use, texs, 224, 224)  # rendered_images: batch * 3 * h * w, masks: batch * h * w
            #     imgtemp = rendered_images.data[0].cpu().numpy()
            #     imgtemp = (np.transpose(imgtemp, (1,2,0)) * 255).astype(np.int)
            #     imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
            #     imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
            #     imgtemp = imgtemp[:, :, [2,1, 0]]
            #     cv2.imwrite("./imgtemp.jpg", imgtemp)

                # imgtemp = masks.data.cpu().numpy()
                # imgtemp = (np.transpose(imgtemp, (1, 2, 0)) * 255).astype(np.int)
                # imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
                # imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
                # imgtemp = np.repeat(imgtemp, 3, axis=2)
                # cv2.imwrite("./mask.jpg", imgtemp)

                # imgtemp = depths.data.cpu().numpy()
                # imgtemp = np.transpose(imgtemp, (1, 2, 0)).astype(np.int)
                # imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
                # imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
                # imgtemp = np.repeat(imgtemp, 3, axis=2)
                # cv2.imwrite("./depths.jpg", imgtemp)

                # masks_erode = self.generate_erode_mask(masks, kernal_size=15)
                # rendered_images = rendered_images.cpu()
                # if grey_background:
                #     rendered_images_erode = masks_erode * rendered_images
                # else:
                #     inv_masks_erode = (torch.ones_like(masks_erode) - (masks_erode)).float()
                #     if avg_BG:
                #         contentsum = torch.sum(torch.sum(masks_erode * rendered_images, 3), 2)
                #         sumsum = torch.sum(torch.sum(masks_erode, 3), 2)
                #         contentsum[contentsum == 0] = 0.5
                #         sumsum[sumsum == 0] = 1
                #         masked_sum = contentsum / sumsum
                #         masked_BG = masked_sum.unsqueeze(2).unsqueeze(3).expand(rendered_images.size())
                #     else:
                #         masked_BG = 0.5
                #     rendered_images_erode = masks_erode * rendered_images + inv_masks_erode * masked_BG

                # texs_a_crop = []
                # for n in range(bz):
                #     tex_a_crop = self.get_render_from_vertices(rendered_images_erode[n], vertices_in_ori_img[n])
                #     texs_a_crop.append(tex_a_crop)
                # texs = torch.cat(texs_a_crop, 0)

            # render face to rotated pose
            with torch.no_grad():
                with torch.cuda.device(self.current_gpu):
                    rendered_images = self.renderer(vertices, self.faces_use, texs, 256, 256)     # 53215*3， 105840*3， 105840*2*2*2*3
                    # 是Render的过程，但不是render-to-image的过程，还没有对人脸中错误的像素渲染结果进行修正
                imgtemp = rendered_images.data[0].cpu().numpy()
                imgtemp = (imgtemp * 255).astype(np.int)
                imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
                imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
                imgtemp = imgtemp[:, :, [2, 1, 0]]
                cv2.imwrite("./imgtemp_0.jpg", imgtemp)

                # imgtemp = masks.data.cpu().numpy()
                # imgtemp = (np.transpose(imgtemp, (1, 2, 0)) * 255).astype(np.int)
                # imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
                # imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
                # imgtemp = np.repeat(imgtemp, 3, axis=2)
                # cv2.imwrite("./mask_0.jpg", imgtemp)
                #
                # imgtemp = depths.data.cpu().numpy()
                # imgtemp = np.transpose(imgtemp, (1, 2, 0)).astype(np.int)
                # imgtemp = np.where(imgtemp >= 0, imgtemp, 0)
                # imgtemp = np.where(imgtemp <= 255, imgtemp, 255)
                # imgtemp = np.repeat(imgtemp, 3, axis=2)
                # cv2.imwrite("./depths_0.jpg", imgtemp)
            # rendered_images = rendered_images.cpu()
            #
            # # get rendered face vertices
            # texs_b = []
            # for n in range(bz):
            #     tex_b = self.get_render_from_vertices(rendered_images[n], vertices_out[n])
            #     texs_b.append(tex_b)
            # texs_b = torch.cat(texs_b, 0)
            #
            # with torch.cuda.device(self.current_gpu):
            #     # rendered_images_rotate, depths1, masks1, = self.renderer(vertices_ori_normal, self.faces, texs_b)  # rendered_images: batch * 3 * h * w, masks: batch * h * w
            #     rendered_images_double, depths2, masks2, = self.renderer(vertices_aligned_normal, self.faces_use, texs_b)  # rendered_images: batch * 3 * h * w, masks: batch * h * w

        return rendered_images, self.torch_get_68_points(vertices_aligned_out), original_angles, self.torch_get_106_points(vertices_aligned_out)
