import copy
import glob
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

from .data_handlers import WFC3UVISHandler, WFC3IRHandler, NIRCamHandler, MIRIHandler, ImageHandler


def _make_perfect_cd(wcs: WCS) -> np.ndarray:
    """Create a perfect (square, orthogonal, undistorted) CD matrix from the
    input WCS.
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
                wcs_copies.append(WCS(w.to_fits_sip(degree=5)))
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
    flat_inds = np.arange(len(xy_arr)).astype(int)
    print(
        f"Separating {len(xy_arr)} points into {2**n_divisions} blocks to reduce memory usage when clustering"
    )
    index_blocks = _separate(xy_arr, flat_inds, distance_threshold * 3.25, n_divisions)
    block_sizes = [len(block) for block in index_blocks]
    print(f"{len(index_blocks)} blocks at size {block_sizes}")
    print(
        "Clustering points based on position in output frame.  This may take a while."
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
    -----
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
    # Create the output WCS grid based on the provided WCS objects
    print("Creating output WCS grid")
    outwcs = _create_output_wcs(wcs_list)

    # Stack the input catalogs and map positions to the output grid
    big_table = vstack(tbls)
    rx, ry = outwcs.world_to_pixel_values(big_table[ra_key], big_table[dec_key])
    big_table["x"] = rx
    big_table["y"] = ry

    arr = np.array([rx, ry]).T

    # Cluster objects based on their positions in the output frame
    # Consider adding cluster sizing as an additional parameter
    print("Clustering points based on position in output frame. This may take a while.")
    labels = separate_and_agglomerate(arr, match_size, 8)
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
    n_dets = [len(g) for g in grptbl.groups]

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


def _find_catalogs(image: str) -> list[list[Table]]:
    """For a given image, return a list of Tables for catalogs for each sci extension"""
    cat_root = image.replace(".fits", "")
    cat_str = f"{cat_root}_sci?_xyrd.cat"
    tbls = []
    for tfile in sorted(glob.glob(cat_str)):
        tmp = Table.read(tfile, format="ascii.commented_header")
        tmp.meta["sci_ext"] = int(tfile.split("_")[-2].replace("sci", ""))
        tbls.append(tmp)
    return tbls


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
    -----
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
        "MIRI/MIRIMAGE": MIRIHandler
    }

    # Select the appropriate ImageHandler class based on instrument and detector
    ih_type = ih_dict[f"{inst}/{det}"]

    # Initialize the selected ImageHandler class with the image, catalog, and metadata
    return ih_type(image, catalog, catalog.meta["sci_ext"])


def make_final_catalog(images: list[str], catalogs: list = []) -> Table:
    if not catalogs:
        catalogs = [_find_catalogs(image) for image in images]

    image_list = []
    print(images, catalogs)
    for image, imcats in zip(images, catalogs):
        for imcat in imcats:
            image_list.append(_to_imagehandler(image, imcat))

    for handler in image_list:
        handler.correct_catalog(radius=10.0)
    tbls = [handler.catalog for handler in image_list]
    wcs_list = [handler.wcs for handler in image_list]
    print(wcs_list)
    return merge_catalogs(tbls, wcs_list)


# ----------------------------Coverage Utilities------------------------


def compute_coverage(cat: Table, ref_wcs: WCS, wcs_list: list) -> np.array:
    """
    This function computes how many exposures and seconds cover a
    catalog of positions in a drizzled frame.

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
    """Remove points outside of RA/DEC bounding box to prevent Nonconvergence error"""
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
            return WCS(w.to_fits_sip(degree=5))
        else:
            raise TypeError(f"Unrecognized format for WCS type {type(w)}.")
    else:
        return w
