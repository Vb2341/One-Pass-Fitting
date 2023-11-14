"""
This module contains the utilities for merging catalogs from many images into one 
master catalog.  Specifically, it contains utilities to create an undistored output
WCS frame that fits all of the input images, project all of the input catalogs' 
positions into that frame and then cross match across all of those catalogs 
simultaneously.  After cross matching, it takes each group of cross matched stars 
and computes the mean and standard deviation of the fluxes and positions.
"""

import copy
import os
from itertools import product

import gwcs
import numpy as np
import tqdm
from astropy.io import fits
from astropy.table import Table, join, vstack
from astropy.wcs import WCS
from sklearn.cluster import AgglomerativeClustering
from stwcs import distortion
from typing import Union

from .data_handlers import (
    WFC3UVISHandler,
    WFC3IRHandler,
    NIRCamHandler,
    MIRIHandler,
    ImageHandler,
    find_catalogs,
    _read_and_check_tbls
)

__all__ = [
    "make_jwst_tweakreg_catfile",
    "make_final_catalog",
    "merge_catalogs",
    "separate_and_agglomerate",
    "compute_coverage",
]


def _make_perfect_cd(wcs: WCS) -> np.ndarray:
    """
    Create a perfect (square, orthogonal, undistorted) CD matrix from the input WCS.
    """
    def_scale = (wcs.pscale) / 3600.0
    def_orientat = np.deg2rad(wcs.orientat)
    perfect_cd = def_scale * np.array(
        [
            [-np.cos(def_orientat), np.sin(def_orientat)],
            [np.sin(def_orientat), np.cos(def_orientat)],
        ]
    )
    return perfect_cd


def _create_output_wcs(wcs_list: list[Union[WCS, gwcs.wcs.WCS]]) -> WCS:
    """
    Create an output World Coordinate System (WCS) for a given list of input WCS objects.

    This function takes a list of input WCS objects and processes them to create an output WCS
    that is suitable for fitting to an output grid. It makes copies of the input WCS objects and
    ensures they are in the proper format. If an input WCS is of an unrecognized format or type,
    an error is raised.

    Parameters:
    -----------
    wcs_list : list
        A list of WCS objects, which may include Astropy WCS or compatible WCS objects.

    Returns:
    --------
    astropy.wcs.WCS
        An output WCS that fits the desired output grid.

    Example
    -------
    wcs_list = [wcs1, wcs2, wcs3]  # List of input WCS objects
    output_wcs = _create_output_wcs(wcs_list)
    # `output_wcs` contains the output WCS suitable for fitting to the output grid.

    Notes
    -----
    - This function processes input WCS objects, making copies and ensuring they are in the
      appropriate format.
    - It is particularly useful when combining or aligning multiple input WCS objects to create
      an output WCS for use in astrometric transformations.
    - If an unrecognized WCS format is encountered, a `TypeError` is raised.
    """
    # make copies of the WCS objects
    wcs_copies = []
    for i, w in enumerate(wcs_list):
        # Need to make sure each is an astropy WCS or child class?
        if not issubclass(WCS, type(w)):
            # Convert GWCS to astropy WCS, it'll be close enough for this step
            if isinstance(w, gwcs.wcs.WCS):
                wcs_copies.append(WCS(w.to_fits_sip(degree=9)))
            else:
                raise TypeError(
                    f"Unrecognized format for WCS at index {i}, \
                                type is {type(w)}."
                )
        else:
            wcs_copies.append(copy.deepcopy(w))
    # actually create and return output WCS that fits output grid
    outwcs = distortion.utils.output_wcs(wcs_copies)
    outwcs.wcs.cd = _make_perfect_cd(outwcs)
    return outwcs


def _find_gaps(vals, min_width=3):
    """
    Find the midpoint of the gap closest to the median in an array of values.

    Parameters:
    -----------
    vals : numpy.ndarray
        An array of values.

    min_width : int, optional
        The minimum width of a gap to be considered.

    Returns:
    --------
    float
        The midpoint of the gap closest to the median, or -1 if no suitable gap is found.

    Notes
    -----
    - This function identifies the midpoint of the closest to median gap in an array of values,
      considering only gaps with a width greater than or equal to the specified `min_width`.
    - If no suitable gap is found, it returns -1.
    """
    # Sort the input values in ascending order
    svals = np.sort(vals)

    # Calculate the differences between adjacent values
    sdiffs = np.diff(svals)

    # Calculate the midpoints of the gaps
    gap_mids = svals[:-1] + sdiffs / 2.0

    # Calculate the index corresponding to the midpoint of the input array
    mid_ind = int(len(vals) / 2.0)

    # Create a mask for gaps that meet the minimum width requirement
    gmask = sdiffs >= min_width

    # Find the indices of the qualifying gaps
    ginds = np.where(gmask)[0]

    # If no qualifying gap is found, return -1
    if len(ginds) == 0:
        return -1

    # Calculate the distances from the midpoint index to the gap indices
    midpt_distance = np.abs(ginds - mid_ind)

    # Find the index of the closest qualifying gap
    closest = ginds[np.argmin(midpt_distance)]

    # Return the midpoint of the closest qualifying gap
    return gap_mids[closest]


def _separate(xy_arr, inds, min_width, depth):
    """Recursively partition xy_arr into subregions"""
    if (depth <= 0) or (len(inds) < 5000):
        return [inds]
    else:
        print("-" * 72)
        bals = []
        pts = []
        for i in range(xy_arr.shape[-1]):
            split_pt = _find_gaps(xy_arr[:, i], min_width)
            lm = xy_arr[:, i] < split_pt
            bals.append(np.abs(np.sum(lm) - np.sum(~lm)))
            pts.append(split_pt)

        dir_ind = np.argmin(bals)
        split = pts[dir_ind]
        left_mask = xy_arr[:, dir_ind] < split

        if split == -1:
            print("BROKEN")
        else:
            print(
                f"SUCCESS {depth}, splitting in axis {dir_ind} with {np.sum(left_mask)} {np.sum(~left_mask)}"
            )

        inds_left = _separate(xy_arr[left_mask], inds[left_mask], min_width, depth - 1)
        inds_right = _separate(
            xy_arr[~left_mask], inds[~left_mask], min_width, depth - 1
        )

        return inds_left + inds_right


def _agglomerate(
    xy_sub: np.ndarray, distance_threshold: float = 0.5, **kwargs
) -> np.ndarray:
    """Actually runs the agglomeration and returns the cluster label for each point"""
    agg = AgglomerativeClustering(
        distance_threshold=distance_threshold, n_clusters=None, **kwargs
    )
    clust = agg.fit(xy_sub)
    return clust.labels_


def separate_and_agglomerate(
    xy_arr: np.ndarray, distance_threshold: float = 0.5, n_divisions: int = 3, **kwargs
) -> np.ndarray:
    """
    Separates and agglomerates data points into clusters based on distance in a multi-step process.

    This function takes an array of 2D data points, divides them into smaller blocks to reduce memory usage,
    and then clusters the points within each block. It returns an array of labels indicating the cluster
    assignment of each data point.

    Parameters:
    -----------
    xy_arr : numpy.ndarray
        An array of data points with two columns (x and y coordinates).

    distance_threshold : float, optional
        The distance threshold for clustering data points (default is 0.5).

    n_divisions : int, optional
        The number of divisions to reduce memory usage (default is 3). A higher value creates smaller blocks.

    **kwargs : keyword arguments
        Additional keyword arguments to be passed to the AgglomerativeClustering object.

    Returns:
    --------
    numpy.ndarray
        An array of labels indicating the cluster assignment of each data point.

    Notes:
    ------
    The 'separate_and_agglomerate' function is a multi-step clustering process. It first divides the input
    data points into smaller blocks to reduce memory usage. These blocks are processed separately and then
    combined into a final clustering result.

    This function prints informative messages about its progress, such as the number of data points and blocks
    being processed. The progress is displayed using the tqdm library.

    For each block, the function clusters the data points using the AgglomerativeClustering with the provided
    distance threshold and any additional keyword arguments.

    The function returns an array of labels, with each label indicating the cluster assignment of the corresponding
    data point.
    """
    flat_inds = np.arange(len(xy_arr)).astype(int)
    print(
        f"Separating {len(xy_arr)} points into {2**n_divisions} blocks to reduce memory usage when clustering"
    )
    index_blocks = _separate(xy_arr, flat_inds, distance_threshold * 3.25, n_divisions)
    block_sizes = [len(block) for block in index_blocks]
    print(f"{len(index_blocks)} blocks at size {block_sizes}")
    print(
        "Clustering points based on position in the output frame. This may take a while."
    )
    start = 0
    labels = np.zeros(len(xy_arr)).astype(int)
    for block in tqdm.tqdm(index_blocks):
        xy_sub = xy_arr[block]
        sub_labels = _agglomerate(xy_sub, distance_threshold, **kwargs)
        labels[block] = sub_labels + start
        start = np.amax(labels) + 1

    return labels


def merge_catalogs(
    tbls: list[Table],
    wcs_list: list[Union[WCS, gwcs.wcs.WCS]],
    match_size: float = 0.5,
    ra_key: str = "RA",
    dec_key: str = "Dec",
    ref_wcs=None,
) -> Table:
    """
    Merge and aggregate multiple catalogs based on their positions in the output frame.

    This function merges multiple catalogs of objects by matching their positions in the output frame.
    It clusters the objects based on their positions and computes aggregate statistics (mean and standard
    deviation) for each cluster.

    Parameters:
    -----------
    tbls : list of astropy.table.Table
        A list of tables, each containing the psf photometry table.

    wcs_list : list of either astropy.wcs.WCS, gwcs.wcs.WCS
        A list of WCS objects representing the world coordinate systems of the images
        corresponding to each table in `tbls`.

    match_size : float, optional
        The maximum distance that objects within a cluster can be apart, matching threshold.

    ra_key : str, optional
        The column key for Right Ascension (RA) in the catalog tables.

    dec_key : str, optional
        The column key for Declination (Dec) in the catalog tables.

    Returns:
    --------
    astropy.table.Table
        A table containing the merged and aggregated catalog.

    Notes:
    ------
    - This function creates an output WCS grid based on the provided WCS objects.
    - It projects all the positions into pixel positions in the output WCS frame
    - Clustering of objects is based on their positions using the `separate_and_agglomerate` function.
    - The function computes mean and standard deviation values for each cluster.
    - The returned table contains the merged and aggregated catalog.
    

    Example:
    --------
    tbls = [table1, table2, table3]
    wcs_list = [wcs1, wcs2, wcs3]
    match_size = 0.5
    merged_catalog = merge_catalogs(tbls, wcs_list, match_size)
    # `merged_catalog` contains the merged and aggregated catalog.
    """
    if ref_wcs is None:
        # Create the output WCS grid based on the provided WCS objects
        print("Creating output WCS grid")
        outwcs = _create_output_wcs(wcs_list)
    else:
        outwcs = ref_wcs
    for t in tbls:
        rx, ry = outwcs.world_to_pixel_values(t[ra_key], t[dec_key])
        t["x"] = rx
        t["y"] = ry
    # Stack the input catalogs and map positions to the output grid
    big_table = vstack(tbls)
    # rx, ry = outwcs.all_world2pix(np.array([big_table[ra_key], big_table[dec_key]]).T, 0).T
    # big_table["x"] = rx
    # big_table["y"] = ry

    arr = np.array([big_table["x"], big_table["y"]]).T

    # Cluster objects based on their positions in the output frame
    # Consider adding cluster sizing as an additional parameter
    print("Clustering points based on position in output frame. This may take a while.")
    labels = separate_and_agglomerate(arr, match_size, 3)
    big_table["label"] = labels

    # Report the number of groups found and the total number of measurements
    print(f"Found {len(set(labels))} groups from {len(big_table)} measurements.")

    # Group the stacked table by cluster label and compute aggregate statistics
    # Doing this grouping creates a small sub table for each star
    grptbl = big_table.group_by("label")
    print("Collating matches, computing means")
    meantbl = grptbl.groups.aggregate(np.nanmean)
    print("Computing standard deviations")
    stdtbl = grptbl.groups.aggregate(np.nanstd)
    # n_dets = [len(g) for g in grptbl.groups]
    n_dets = grptbl.groups.aggregate(len)["m"]

    # Add metadata names for mean and standard deviation tables
    meantbl.meta["name"] = "mean"
    stdtbl.meta["name"] = "std"

    # Join the mean and standard deviation tables and add the number of detections
    res = join(
        meantbl, stdtbl, join_type="left", keys="label", table_names=["mean", "std"]
    )
    res["n"] = n_dets
    res["n_expected"] = compute_coverage(res, outwcs, wcs_list)

    return res





def _to_imagehandler(image: str, catalog: Table) -> ImageHandler:
    """
    Create an ImageHandler instance based on the instrument and detector in the image header.

    This function takes an image file and a catalog of objects, reads the image header to
    determine the instrument and detector, and then selects the appropriate ImageHandler subclassclass
    based on this information. It returns an instance of the selected ImageHandler subclass.

    Parameters:
    -----------
    image : str
        The path to the FITS image file.

    catalog : astropy.table.Table
        A table containing the photometry catalog of the image. The metadata dictionary
        must contain a key `sci_ext` reflecting the science extension number that `catalog`
        is derived from

    Returns:
    --------
    ImageHandler
        An instance of the appropriate ImageHandler subclass for the given instrument and detector.

    Notes:
    ------
    - The function determines the instrument and detector from the image header.
    - Depending on the instrument and detector, it selects the appropriate ImageHandler class.
    - The selected ImageHandler class is initialized with the image file, catalog, and the
      metadata associated with the scientific extension.

    """
    # Read the image header to obtain instrument and detector information
    hdr = fits.getheader(image)
    inst = hdr["INSTRUME"]
    det = hdr["DETECTOR"]

    # Handle special case for NIRCAM instrument
    if inst == "NIRCAM":
        det = "NRC"

    # Dictionary to map instrument/detector combinations to ImageHandler classes
    ih_dict = {
        "WFC3/UVIS": WFC3UVISHandler,
        "WFC3/IR": WFC3IRHandler,
        "NIRCAM/NRC": NIRCamHandler,
        "MIRI/MIRIMAGE": MIRIHandler,
    }

    # Select the appropriate ImageHandler class based on instrument and detector
    ih_type = ih_dict[f"{inst}/{det}"]

    # Initialize the selected ImageHandler class with the image, catalog, and metadata
    return ih_type(image, catalog)


def make_final_catalog(
    images: list[str],
    catalogs: list[list] = [],
    match_size: float = 0.5,
    ra_key: str = "RA",
    dec_key: str = "Dec",
    ref_wcs=None,
) -> Table:
    """
    Create a final catalog by combining multiple input catalogs from a list of images.

    This function takes a list of image filenames and, optionally, a list of corresponding input catalogs.
    It processes the images and their catalogs, applies necessary corrections, and combines them into a
    final catalog. If no input catalogs are provided, the function will attempt to find them using the
    `find_catalogs` function.

    Parameters:
    -----------
    images : list of str
        A list of image filenames for which to create a final catalog.

    catalogs : list of lists of astropy.table.Table, optional
        A list of input catalogs corresponding to the images (default is an empty list).

    match_size : float, optional
        The maximum distance in pixels that objects within a cluster can be apart, matching threshold.

    ra_key : str, optional
        The column key for Right Ascension (RA) in the catalog tables.

    dec_key : str, optional
        The column key for Declination (Dec) in the catalog tables.

    Returns:
    --------
    astropy.table.Table
        A final catalog that combines information from the input catalogs of all the provided images.

    Notes:
    ------
    This function processes a list of images and their corresponding catalogs, and produces a final catalog
    that combines the information from all input catalogs. It first checks if input catalogs are provided; if not,
    it uses the `find_catalogs` function to attempt to locate them.  `find_catalogs` requires the catalogs
    to be named specifically with the suffix `_sci<X>_xyrd.cat` replacing the .fits (where <X> corresponds to the number
    of the science extension, i.e. `idny01a1q_flc_sci1_xyrd.cat` corresponding to extension `idny01a1q_flc.fits[SCI, 1] )

    After loading image catalogs into `ImageHandler` objects, it applies corrections to the catalogs, such as
    aperture corrections, if necessary. Then, it combines all the corrected catalogs into a final output catalog using `merge_catalogs()`.

    The final catalog will contain the averages and standard deviations of the quantities measured for the matched and
    aggregated stars.  See `merge_catalogs()` for details.

    Example:
    --------
    To create a final catalog from a list of images, you can use this function as follows:

    >>> image_files = ["image1.fits", "image2.fits"]
    # Note how one image can have multiple SCI extensions, and thus multiple catalogs (e.g. WFC3/UVIS)
    >>> catalogs = [["image1_sci1_xyrd.cat", "image1_sci2_xyrd.cat"], ["image2_sci1_xyrd.cat"]]
    >>> final_catalog = make_final_catalog(images=image_files, catalogs=catalogs, match_size=0.5)
    """

    if not catalogs:
        catalogs = [find_catalogs(image) for image in images]
    elif len(catalogs) == len(images):
        catalogs = [_read_and_check_tbls(cats) for cats in catalogs]

    image_list = []
    # print(images, catalogs)
    for image, imcats in zip(images, catalogs):
        for imcat in imcats:
            image_list.append(_to_imagehandler(image, imcat))

    for handler in image_list:
        handler.correct_catalog(aperture=False, pixel_area=False, radius=10.0)
    tbls = [handler.catalog for handler in image_list]
    wcs_list = [handler.wcs for handler in image_list]
    # print(wcs_list)
    return merge_catalogs(tbls, wcs_list, match_size, ra_key, dec_key, ref_wcs)


def make_jwst_tweakreg_catfile(images, catalogs, catfile_name='tweakreg_catfile.txt'):
    """
    Create a catalog file associating images with their corresponding catalogs, for use in JWST TweakRegStep.

    Parameters
    ----------
    images : list of str
        List of image filenames.
    catalogs : list of Table
        List of Astropy tables representing the catalogs for each image (catalogs myst be in same order as images).
    catfile_name : str, optional
        Name of the output catalog file. Default is 'tweakreg_catfile.txt'.

    Returns
    -------
    catfile_name : str
        The name of the created catalog file.

    Notes
    -----
    This function creates a catalog file that associates image filenames with their
    corresponding catalogs. It writes the catalog file in a format with commented headers.
    Ensure that the catalogs have columns labeled 'x' and 'y' containing the PSF fit pixel positons (0 indexed)!

    Example
    -------
    >>> # catalog1 and catalog2 are astropy Tables
    >>> catfile = make_catfile(['image1.fits', 'image2.fits'], [catalog1, catalog2], 'my_catfile.txt')
    >>> print(f'Created catalog file: {catfile}')
    """

    catnames = []  # Initialize a list to store catalog filenames

    for im, cat in zip(images, catalogs):
        catname = im.replace('.fits', "_sci1_xyrd.ecsv")  # Generate catalog filename
        catnames.append(catname)  # Append catalog filename to the list
        cat.write(catname, overwrite=True)  # Write the catalog to the file

    # Create an Astropy table associating images with catalog filenames
    images = [os.path.split(image)[-1] for image in images] # catfile must have just the image filename, not with the relative path
    catfile_tbl = Table([images, catnames], names=['image', 'catalog'])

    # Write the table to the catalog file in a format with commented headers
    catfile_tbl.write(catfile_name, format='ascii.commented_header', overwrite=True)

    print(f'Wrote tweakreg catfile to {catfile_name}')  # Print a confirmation message

    return catfile_name  # Return the name of the created catalog file

# ----------------------------Coverage Utilities------------------------


def compute_coverage(cat: Table, ref_wcs: WCS, wcs_list: list) -> np.array:
    """Computes how many exposures and seconds cover a catalog of positions in a drizzled frame.

    For every position in a catalog corresponding to a drizzled frame,
    this function returns how many exposures covered that point, and
    how many seconds of exposure time covered that point.  It does this
    by reconstructing the WCS of every input exposure/extension, and
    transforming the positions in the catalog to the exposure/extension
    frame, and seeing if the transformed positions land on the array.
    It then adds up the exposure time of each exposure, accounting for
    each exposures coverage of the different positions.


    Parameters
    ----------
    cat : astropy Table
        Catalog containing X and Y positions in drizzled frame

    Returns
    -------
    n_exp : numpy array
        Number of exposures that cover the X and Y positions in `cat`
    """

    rds = np.array(ref_wcs.pixel_to_world_values(cat["x_mean"], cat["y_mean"]))

    wcs_list
    arr = np.zeros((len(wcs_list), len(cat)))
    for i, iwcs in enumerate(wcs_list):
        _w = _convert_wcs(iwcs)
        arr[i] = _transform_points(rds, _w)
    n_exp = np.sum(arr, axis=0).astype(int)
    # if 'EXPTIME' in hdrtab.colnames:
    #     etime = np.sum(arr*np.array(hdrtab['EXPTIME'])[:,None], axis=0)
    # else:
    #     etime = np.sum(arr*np.array(hdrtab['DURATION'])[:,None], axis=0)

    return n_exp


def _prefilter_coordinates(input_skycoords, xmin, xmax, ymin, ymax, flt_wcs):
    """
    Remove points outside of RA/DEC bounding box to prevent Nonconvergence error

    Sometimes when projecting sky coordinates onto a pixel grid, if the sky coordinates
    are too far of the pixel array, the function crashes.  This function calculates the
    bounding box of the pixel array in sky coordinates, and throws out any of the input
    sky coordinates that are outside of the box.
    """
    # this probably will explode if the image contains a celestial pole?

    inp_ra, inp_dec = np.array(input_skycoords)

    corners = np.array([p for p in product([xmin, xmax], [ymin, ymax])])
    corner_ra, corner_dec = flt_wcs.pixel_to_world_values(*corners.T)

    mask_ra = (inp_ra > np.amin(corner_ra)) & (inp_ra < np.max(corner_ra))
    mask_dec = (inp_dec > np.amin(corner_dec)) & (inp_dec < np.amax(corner_dec))
    mask = mask_ra & mask_dec
    return mask
    # return input_skycoords[mask]


def _transform_points(
    input_skycoords: np.ndarray, flt_wcs: WCS, padding: float = 0.0
) -> np.ndarray:
    """
    Checks if a sky coordinate falls within the footprint of an image.

    Parameters:
    -----------
    input_skycoords : ndarray
        An Nx2 array of ra and dec coordinates

    flt_wcs : astropy.wcs.WCS
        The World Coordinate System (WCS) for the target frame.

    padding : int, optional
        Padding to extend the valid pixel region in the frame.

    Returns:
    --------
    on_frame : ndarray
        An array of binary flags indicating whether each input coordinate is within the frame.

    Notes
    -----
    - This function transforms celestial coordinates to pixel coordinates using the provided WCS.
    - The `padding` parameter allows extending the valid pixel region.
    - The `on_frame` output contains binary flags where 1 indicates a coordinate is within the frame.
    """
    xmin = ymin = -1 * padding
    if hasattr(flt_wcs, "_naxis"):
        naxis = flt_wcs._naxis
    elif isinstance(flt_wcs, gwcs.wcs.WCS):
        naxis = WCS(flt_wcs.to_fits_sip())._naxis

    xmax, ymax = np.array(naxis) + padding

    bboxmask = _prefilter_coordinates(input_skycoords, xmin, xmax, ymin, ymax, flt_wcs)
    inds = np.arange(len(input_skycoords.T)).astype(int)
    filtered_coords = input_skycoords.T[bboxmask]
    bbinds = inds[bboxmask]
    xc, yc = flt_wcs.world_to_pixel_values(*filtered_coords.T)
    mask = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)

    good_inds = bbinds[mask]
    on_frame = np.zeros(inds.shape).astype(int)
    on_frame[good_inds] = 1

    return on_frame


def _convert_wcs(w: Union[WCS, gwcs.wcs.WCS]) -> WCS:
    """Converts WCS to a FITSWCS if not already"""
    if not issubclass(WCS, type(w)):
        # Convert GWCS to astropy WCS, it'll be close enough for this step
        if isinstance(w, gwcs.wcs.WCS):
            return WCS(w.to_fits_sip(degree=9))
        else:
            raise TypeError(f"Unrecognized format for WCS type {type(w)}.")
    else:
        return w
