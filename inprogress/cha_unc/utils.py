import numpy
import numpy as np
from joblib import dump, load
import scipy.stats
import matplotlib.pyplot as plt
import branca.colormap as cm

import matplotlib

matplotlib.use('Agg')


def reversed_colormap(existing):
    return cm.LinearColormap(colors=list(reversed(existing.colors)),
                             vmin=existing.vmin, vmax=existing.vmax)


def colormap_alpha(existing):
    '''Adds alpha channel from 1 to 0 to existing colormap'''
    new_colors = []
    for i, (r, g, b, a) in enumerate(existing.colors):
        new_a = i/(len(existing.colors)-1)
        new_colors.append((r,g,b,new_a))
    return cm.LinearColormap(colors=new_colors,
                             vmin=existing.vmin, vmax=existing.vmax)

def load_data():
    x = numpy.load('./data/x.npy')
    y = numpy.load('./data/y.npy')

    # Remove overlaping predictions to make the visualisations clearer
    x, tmp_idx = numpy.unique(x, axis=0, return_index=True)
    y = y[tmp_idx]

    # Remove very close point in Westbury going down
    westbury_coordinates = [51.49354491805622, -2.618626674263542]
    closer_idx = np.where((np.sum(np.abs(x - [51.49354491805622, -2.618626674263542]), axis=1) < 0.01) & y==1)[0]
    x = np.delete(x, closer_idx, axis=0)
    y = np.delete(y, closer_idx)

    return x, y

from ipyleaflet import Map, Heatmap

def ipyleaflet_heatmap_per_class(x, y, center=(0, 0), zoom=5, colors=['blue', 'red', 'yellow', 'orange']):
    all_classes = np.unique(y)
    m = Map(center=center, zoom=zoom)

    for i in all_classes:
        heatmap = Heatmap(
            locations=x[y==i].tolist(),
            radius=5,
            gradient={0.0: colors[i], 1.0: colors[i]},
            min_opacity=1,
        )

        m.add_layer(heatmap)
    return m

from ipyleaflet import Map, basemaps, basemap_to_tiles, Circle
from sklearn.utils import shuffle

def ipyleaflet_scatterplot_per_class(x, y, center=(0, 0), zoom=5, proportion=1.0,
                                     colors=['blue', 'red', 'yellow', 'orange'],
                                     m=None, width='600px', height='400px'):
    if m is None:
        m = Map(center=center, zoom=zoom)

    m.layout.width = width
    m.layout.height = height

    if proportion < 1.0:
        n_samples = int(x.shape[0]*proportion)
        x, y = shuffle(x, y)
        x = x[:n_samples]
        y = y[:n_samples]

    for sample_x, sample_y in zip(x.tolist(), y.tolist()):
        circle = Circle()
        circle.location = sample_x
        circle.radius = 100
        circle.color = colors[sample_y]
        circle.fill_color = colors[sample_y]

        m.add_layer(circle)
    return m

from sklearn.neighbors import KernelDensity

class KDE:
    def __init__(self, bandwidth=0.01):
        self.bandwidth=bandwidth
        self.pdf_0 = None
        self.pdf_1 = None
        self.pi = None
    def fit(self, x, y):
        self.pi = numpy.array([numpy.sum(1 - y), numpy.sum(y)])
        self.pi = self.pi / numpy.sum(self.pi)
        self.pdf_0 = KernelDensity(kernel='exponential', bandwidth=self.bandwidth)
        self.pdf_0.fit(x[y==0, :])
        self.pdf_1 = KernelDensity(kernel='exponential', bandwidth=self.bandwidth)
        self.pdf_1.fit(x[y==1, :])
    def predict(self, x):
        joint = numpy.hstack([(self.pi[0] * numpy.exp(self.pdf_0.score_samples(x))).reshape(-1, 1),
                              (self.pi[1] * numpy.exp(self.pdf_1.score_samples(x))).reshape(-1, 1)])
        return numpy.argmax(joint / numpy.sum(joint, axis=1).reshape(-1, 1), axis=1)
    def predict_proba(self, x):
        n = numpy.shape(x)[0]
        joint = numpy.hstack([(self.pi[0] * numpy.exp(self.pdf_0.score_samples(x))).reshape(-1, 1),
                              (self.pi[1] * numpy.exp(self.pdf_1.score_samples(x))).reshape(-1, 1)])
        return joint / numpy.sum(joint, axis=1).reshape(-1, 1)


def mpl_scatter_contourlines(clf, lat_grid, lon_grid, x, y, xlim=None, ylim=None,
                             isolines=np.linspace(0,1,10)):
    '''Plot and return contourlines'''
    probabilities = clf.predict_proba(numpy.hstack([lat_grid.reshape(-1, 1),
                                                    lon_grid.reshape(-1, 1)]))[:, 1].reshape(lat_grid.shape[0],
                                                                                             lon_grid.shape[0])

    fig, ax = plt.subplots(1, figsize=(12, 9))
    # TODO Consider using skimage.measure.find_contours instead
    cs = ax.contour(lat_grid,  lon_grid, probabilities, isolines, alpha=0.8, cmap='coolwarm', linewidths=(3,))
    ax.clabel(cs, fmt='%2.1f', colors='k', fontsize=14)
    fig.colorbar(cs)
    ax.scatter(x[:,0], x[:,1], c=y, edgecolors='k', cmap='bwr')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return fig, ax, cs


def mpl_scatter_contourf(clf, lat_grid, lon_grid, x, y, xlim=None, ylim=None,
                         isolines=np.linspace(0,1,10)):
    '''Plot and return contourmap'''
    probabilities = clf.predict_proba(numpy.hstack([lat_grid.reshape(-1, 1),
                                                    lon_grid.reshape(-1, 1)]))[:, 1].reshape(lat_grid.shape[0],
                                                                                             lon_grid.shape[0])

    fig, ax = plt.subplots(1, figsize=(12, 9))
    cs = ax.contourf(lat_grid,  lon_grid, probabilities, isolines, alpha=0.8, cmap='bwr')
    fig.colorbar(cs)
    ax.scatter(x[:,0], x[:,1], c=y, edgecolors='k', cmap='bwr')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return fig, ax, cs

def split_contours(segs, kinds=None):
    """takes a list of polygons and vertex kinds and separates disconnected vertices into separate lists.
    The input arrays can be derived from the allsegs and allkinds atributes of the result of a matplotlib
    contour or contourf call. They correspond to the contours of one contour level.

    Example:
    cs = plt.contourf(x, y, z)
    allsegs = cs.allsegs
    allkinds = cs.allkinds
    for i, segs in enumerate(allsegs):
        kinds = None if allkinds is None else allkinds[i]
        new_segs = split_contours(segs, kinds)
        # do something with new_segs

    More information:
    https://matplotlib.org/3.3.3/_modules/matplotlib/contour.html#ClabelText
    https://matplotlib.org/3.1.0/api/path_api.html#matplotlib.path.Path
    Source:
    https://stackoverflow.com/questions/65634602/plotting-contours-with-ipyleaflet
    """
    if kinds is None:
        return segs    # nothing to be done
    # search for kind=79 as this marks the end of one polygon segment
    # Notes:
    # 1. we ignore the different polygon styles of matplotlib Path here and only
    # look for polygon segments.
    # 2. the Path documentation recommends to use iter_segments instead of direct
    # access to vertices and node types. However, since the ipyleaflet Polygon expects
    # a complete polygon and not individual segments, this cannot be used here
    # (it may be helpful to clean polygons before passing them into ipyleaflet's Polygon,
    # but so far I don't see a necessity to do so)
    new_segs = []
    for i, seg in enumerate(segs):
        segkinds = kinds[i]
        boundaries = [0] + list(np.nonzero(segkinds == 79)[0])
        for b in range(len(boundaries)-1):
            new_segs.append(seg[boundaries[b]+(1 if b>0 else 0):boundaries[b+1]])
    return new_segs

import ipyleaflet
from branca.colormap import linear
from ipyleaflet import Map, LegendControl, Polyline
from ipyleaflet import Polygon

def ipyleaflet_contourmap(center, datapoints=None,
                          contourmap=None,
                          isolines=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                          lineopacity=1.0, colormap=linear.viridis,
                          fillopacity=0.7, legend_title='Legend',
                          m=None, zoom=11, width='600px',
                          height='400px'):

    if m is None:
        m = Map(center=center, zoom=zoom)

    m.layout.width = width
    m.layout.height = height

    cs = contourmap
    colors = [colormap(i/(len(cs.levels)-1)) for i in range(len(cs.levels)-1)]
    allsegs = cs.allsegs
    allkinds = cs.allkinds

    for clev in range(len(cs.allsegs)):
        kinds = None if allkinds is None else allkinds[clev]
        segs = split_contours(allsegs[clev], kinds)
        polygons = Polygon(
                        locations=[p.tolist() for p in segs],
                        # locations=segs[14].tolist(),
                        color=colors[clev],
                        weight=2,
                        opacity=lineopacity,
                        fill_color=colors[clev],
                        fill_opacity=fillopacity
        )
        m.add_layer(polygons);

    if datapoints is not None:
        m = ipyleaflet_scatterplot_per_class(datapoints[0], datapoints[1],
                                             proportion=1.0, m=m)

    legend_colors = {}
    for i in reversed(range(len(isolines)-1)):
        legend_colors["{:0.1f}-{:0.1f}".format(isolines[i], isolines[i+1])] = colormap(i/(len(isolines)-1))

    legend = LegendControl(legend_colors, name=legend_title, position="topright")
    m.add_control(legend)
    return m


class AIUKSlides(object):
    def __init__(self, local_center=(51.4545, -2.5879), width='600px',
                 height='400px', isolines=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                 grid_density=500):
        self.local_center = local_center
        self.width = width
        self.height = height
        self.isolines = isolines

        self.x, self.y = load_data()
        self.local_idx = (numpy.sqrt((self.x[:, 0] - local_center[0]) ** 2
                                     + (self.x[:, 1] - local_center[1]) ** 2) <= 0.1)

        self.x_local = self.x[self.local_idx, :]
        self.y_local = self.y[self.local_idx]

        self.x_back = self.x[~self.local_idx, :]
        self.y_back = self.y[~self.local_idx]

        self.xlim = (self.x_local[:,0].min(), self.x_local[:,0].max())
        self.ylim = (self.x_local[:,1].min(), self.x_local[:,1].max())

        self.lon_all_grid = numpy.linspace(-1.75, -3.45, grid_density)
        self.lat_all_grid = numpy.linspace(51.1, 51.8, grid_density)
        self.lat_all, self.lon_all = numpy.meshgrid(self.lat_all_grid, self.lon_all_grid)

    def map_covid_uk(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        return ipyleaflet_scatterplot_per_class(self.x, self.y,
                                                center=(53, -2), zoom=5.5,
                                                proportion=0.1,
                                                width=width, height=height)

    def map_covid_local(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        return ipyleaflet_scatterplot_per_class(self.x[self.local_idx],
                                                self.y[self.local_idx],
                                                center=self.local_center,
                                                zoom=11, proportion=1.0,
                                                width=width, height=height)

    def train_local_classifier(self, clf):
        self.clf = clf
        self.clf.fit(self.x_local, self.y_local)

    def train_local_foreground(self):
        ''' 
            TODO: Use only background check with priors
        '''
        from sklearn.mixture import GaussianMixture

        self.bc = BackgroundCheck(model_foreground=GaussianMixture(n_components=4, covariance_type='spherical'),
                             model_background=GaussianMixture(n_components=32, covariance_type='spherical'))

        self.bc.fit(self.x_local, self.x_back)


    def map_local_classifier(self, zoom=11, fillopacity=0.5, lineopacity=0.5,
                             isolines=None, width=None, height=None):

        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if isolines is None:
            isolines = self.isolines

        # TODO Can we incorporate this call into ipyleaflet_contourmap without automatic ploting?
        fig, ax, contourmap = mpl_scatter_contourf(self.clf, self.lat_all,
                                                   self.lon_all, self.x_local,
                                                   self.y_local,
                                                   isolines=isolines)


        m = ipyleaflet_contourmap(center=self.local_center,
                                  datapoints=[self.x_local, self.y_local],
                                  contourmap=contourmap, isolines=isolines,
                                  colormap=reversed_colormap(linear.RdBu_05),
                                  legend_title='Up vs Down', zoom=zoom,
                                  fillopacity=fillopacity,
                                  lineopacity=lineopacity,
                                  width=width, height=height)
        return m


    def map_local_classifier_foreground(self, zoom=11, fillopacity=0.5, lineopacity=0.5,
                                        isolines=None, width=None,
                                        height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if isolines is None:
            isolines = self.isolines

        prob_per_class = self.clf.predict_proba(numpy.hstack([self.lat_all.reshape(-1, 1),
                                                              self.lon_all.reshape(-1, 1)]))

        p1 = prob_per_class[:, 1].reshape(self.lat_all_grid.shape[0],
                                          self.lon_all_grid.shape[0])
        p0 = prob_per_class[:, 0].reshape(self.lat_all_grid.shape[0],
                                          self.lon_all_grid.shape[0])

        p_local = self.bc.predict_proba(numpy.hstack([self.lat_all.reshape(-1, 1),
                                                      self.lon_all.reshape(-1, 1)]))
        p_local = p_local[:, 1].reshape(self.lat_all.shape[0],
                                        self.lon_all.shape[0])

        p1_not_back = p1 * p_local
        p0_not_back = p0 * p_local

        contourmap_c0_fg = plt.contourf(self.lat_all, self.lon_all,  p0_not_back,
                                        isolines)
        contourmap_c1_fg = plt.contourf(self.lat_all, self.lon_all,  p1_not_back,
                                       isolines)

        Alpha_Reds_08 = colormap_alpha(linear.Reds_08)

        m = ipyleaflet_contourmap(center=self.local_center, contourmap=contourmap_c1_fg,
                              isolines=isolines,
                              colormap=Alpha_Reds_08, legend_title='Up',
                              width=width, height=height)

        Alpha_blues_09 = colormap_alpha(linear.Blues_09)
        ipyleaflet_contourmap(center=self.local_center,
                              datapoints=[self.x_local, self.y_local],
                              contourmap=contourmap_c0_fg, isolines=isolines,
                              colormap=Alpha_blues_09, legend_title='Down',
                              m=m, width=width, height=height)
        return m


class BackgroundCheck(object):
    def __init__(self, model_foreground, model_background=None, mu=[.5, .5]):
        '''TODO Need to check code when background is not given'''
        self.model_foreground = model_foreground
        self.model_background = model_background
        self.mu = mu

    def fit(self, x_foreground, x_background=None, pi=[.5, .5]):
        self.model_foreground.fit(x_foreground)
        if x_background is not None and self.model_background is not None:
            self.model_background.fit(x_background)
            self.pi = numpy.array([len(x_background), len(x_foreground)])
            self.pi = self.pi / self.pi.sum()
        else:
            self.d_max = self.model_foreground.score_samples(x_foreground).max()
            self.pi = [.5, .5]

    def predict_proba(self, x):
        d_fore = np.exp(self.model_foreground.score_samples(x))
        if self.model_background is None:
            d_fore = d_fore / max(self.d_max, d_fore.max())
            d_back = (1 - d_fore)*self.mu[0] + d_fore*self.mu[1]
        else:
            d_back = np.exp(self.model_background.score_samples(x))
        p_fore = (self.pi[1] * d_fore) / (self.pi[1] * d_fore + self.pi[0] * d_back)
        return numpy.vstack((1 - p_fore, p_fore)).T
