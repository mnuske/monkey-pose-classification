import numpy as np
import cv2



def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def resize_image(image, max_size=800):
    if max(image.shape[:2]) > max_size:
        scale = float(max_size) / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def plot_tracking(opt, image, tlwh, pose, scores=None, frame_id=0, fps=0.):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    pose = pose.squeeze().numpy()
    scores = scores.squeeze().numpy()
    order = np.argsort(scores)[::-1]
    scores = scores[order]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f' % (frame_id, fps),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    # cv2.putText(im, 'Pose \t Score/n{} \t {}\n{} \t {}\n{} \t {}\n{} \t{}'.format(opt.class_names[order[0]], scores[0], opt.class_names[order[1]], scores[1], opt.class_names[order[2]], scores[2], opt.class_names[order[3]], scores[3]),
    #             (im_w - int(15 * text_scale), int(15 * text_scale)) , cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)
    text = "Pose\n{}\n{}\n{}\n{}".format(opt.class_names[order[0]], opt.class_names[order[1]], opt.class_names[order[2]], opt.class_names[order[3]])
    text2 = "Score\n{:.2f}\n{:.2f}\n{:.2f}\n{:.2f}".format(scores[0], scores[1], scores[2], scores[3])
    # y0, dy = (im_w - int(15 * text_scale), 20)
    uv_top_left = np.array([im_w-250, 20], dtype=float)
    assert uv_top_left.shape == (2,)
    for i, (line, line2) in enumerate(zip(text.split('\n'), text2.split('\n'))):
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=text_scale,
            thickness=text_thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))
        org2 = tuple((uv_bottom_left_i+[150,0]).astype(int))
        cv2.putText(im, line, org, cv2.FONT_HERSHEY_PLAIN, 1, text_thickness)
        cv2.putText(im, line2, org2, cv2.FONT_HERSHEY_PLAIN, 1, text_thickness)
        line_spacing=1.5
        uv_top_left += [0, h * line_spacing]


    x1, y1, w, h = tlwh
    intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
    id_text = '{:s}'.format(opt.class_names[pose])

    _line_thickness = 1 if pose <= 0 else line_thickness
    color = get_color(abs(pose+1))
    color = tuple([int(x) for x in color])
    cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
    # cv2.putText(im, id_text, (intbox[0], intbox[1] + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
    #             thickness=text_thickness)

    cv2.putText(im,
                opt.class_names[pose],
                (int(x1), int(y1)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 255),  # cls_id: yellow
                thickness=text_thickness)
    return im


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = 'det' if det[5] > 0 else 'trk'
            if ids is not None:
                text = '{}# {:.2f}: {:d}'.format(label, det[6], ids[i])
                cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=thickness)
            else:
                text = '{}# {:.2f}'.format(label, det[6])

        if scores is not None:
            text = '{:.2f}'.format(scores[i])
            cv2.putText(im, text, (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                        thickness=thickness)

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im
