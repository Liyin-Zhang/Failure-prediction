from itertools import cycle
import numpy
from matplotlib import pyplot
from skimage import filters
from PIL import Image
import io
from pathlib import Path

class GANDemoVisualizer:

    def __init__(self, title, l_kde=100, bw_kde=5):
        self.title = title
        self.l_kde = l_kde
        self.resolution = 1. / self.l_kde
        self.bw_kde_ = bw_kde
        self.fig, self.axes = pyplot.subplots(ncols=3, figsize=(13.5, 4)) #13.5 4
        self.fig.canvas.manager.set_window_title(self.title)
        self.frames = []  # 新增：缓存每一帧图像

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=3000, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples, cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples, cmap)

        if show:
            pyplot.draw()
            pyplot.pause(pause_time)

            # === 保存当前帧为图片 ===
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png')
            # self.fig.savefig(f'1',format='png')
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            self.frames.append(image)
            buf.close()

    @staticmethod
    def draw_samples(axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        axis.plot(generated_samples[:max_sample_size, 0], generated_samples[:max_sample_size, 1], '.', label='Generated')
        axis.plot(real_samples[:max_sample_size, 0], real_samples[:max_sample_size, 1], 'kx', label='Real')
        axis.legend(loc='upper right')
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def draw_density_estimation(self, axis, title, samples, cmap):
        axis.clear()
        axis.set_xlabel(title)
        density_estimation = numpy.zeros((self.l_kde, self.l_kde))
        for x, y in samples:
            if 0 < x < 1 and 0 < y < 1:
                density_estimation[int((1 - y) / self.resolution)][int(x / self.resolution)] += 1
        density_estimation = filters.gaussian(density_estimation, self.bw_kde_)
        axis.imshow(density_estimation, cmap=cmap)
        axis.xaxis.set_major_locator(pyplot.NullLocator())
        axis.yaxis.set_major_locator(pyplot.NullLocator())

    def savefig(self, filepath):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)  # 没有就创建
        self.fig.savefig(filepath)

    def save_gif(self, gif_path='fig1.gif', duration=100):
        """保存动画 GIF：每帧保存间隔 `duration` 毫秒"""
        if self.frames:
            self.frames[0].save(
                gif_path,
                save_all=True,
                append_images=self.frames[1:],
                duration=duration,
                loop=0
            )
            print(f"✅ 动图已保存为 {gif_path}")
        else:
            print("⚠️ 没有帧可以保存为 GIF！")

    @staticmethod
    def show():
        pyplot.show()
