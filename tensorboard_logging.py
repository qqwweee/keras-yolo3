"""
Simple example on how to log scalars and images to tensorboard without tensor ops.

License: Copyleft
"""

__author__ = "Michael Gygli"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from io import BytesIO


def log_scalar(callback, tag, value, step):
    """Log a scalar variable.

    Parameter
    ----------
    tag : basestring
        Name of the scalar
    value
    step : int
        training iteration
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                 simple_value=value)])
    callback.writer.add_summary(summary, step)


def log_images(callback, tag, images, step):
    """Logs a list of images."""

    im_summaries = []
    for nr, img in enumerate(images):
        # Write the image to a string
        s = BytesIO()
        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                             image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=im_summaries)
    callback.writer.add_summary(summary, step)


def log_histogram(callback, tag, values, step, bins=1000):
    """Logs the histogram of a list/vector of values."""
    # Convert to a numpy array
    values = np.array(values)

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    callback.writer.add_summary(summary, step)
    callback.writer.flush()