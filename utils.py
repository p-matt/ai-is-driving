import math


def scale_image(img, factor):
    from pygame import transform
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    from pygame import transform
    rotated_image = transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)


def blit_text_center(win, font, text):
    render = font.render(text, 1, (200, 200, 200))
    win.blit(render, (win.get_width() / 2 - render.get_width() / 2, win.get_height() / 2 - render.get_height() / 2))


def get_distance(pos: list):
    # pos: list of (x, y)
    d = sum([math.hypot(x - pos[i + 1][0], y - pos[i + 1][1]) for i, (x, y) in enumerate(pos) if i < len(pos) - 1])
    return d
