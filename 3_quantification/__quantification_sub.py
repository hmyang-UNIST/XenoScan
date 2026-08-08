import math
import skimage
import builtins
import numpy as np


class RegOps():
    """RegOps class performs basic manipulation on region
    ...
    Parameters
    ----------
    img_bw : bool
        binary image
    Attributes
    ----------
    img_bw : bool
        binary image
    h : int
        height
    w : int
        width
    labels : ndarray of dtype int
        labeled array, where all connected regions are assigned the same integer value
    min_row : int
        minimum row of region
    max_row : int
        maximum row of region
    min_col : int
        minimum col of region
    max_col : int
        maximum col of region
    Methods
    -------
    getOrientationCC(self)
        Returns orientation of largest connected component region
    getRowSortedMinMaxAvgCols(self)
        Returns row sorted column values (row, mincol, maxcol, avgcol) ndarray
    getColSortedMinMaxAvgRows(self)
        Returns column sorted row values (col, minrow, maxrow, avgrow) ndarray
    getPolyfitCols(self)
        Returns polynomial fit columns for each row
    getPolyfitRows(self)
        Returns polynomial fit rows for each column
    getPolyfitImg(self)
        Returns bool img with polynomial fit line
    """

    def __init__(self, img_bw):
        self.img_bw = img_bw
        (self.h, self.w) = np.shape(img_bw)
        self.labels = skimage.measure.label(self.img_bw, return_num=False)
        self.largestCC = self.labels == np.argmax(np.bincount(self.labels.flat, weights=self.img_bw.flat))
        self.boundary = skimage.feature.canny(self.largestCC)
        self.min_row, self.max_row = np.min(np.nonzero(self.boundary)[0]), np.max(np.nonzero(self.boundary)[0])
        self.min_col, self.max_col = np.min(np.nonzero(self.boundary)[1]), np.max(np.nonzero(self.boundary)[1])

    def getOrientationCC(self):
        """Returns orientation of largest connected component region
        Parameters
        ----------
        None
        Returns
        -------
        orient : str
            orientation of largest connected component region
        """

        reg = skimage.measure.regionprops(skimage.measure.label(self.largestCC, return_num=False))
        assert len(reg) == 1, "Must be one object only'"
        orient_deg = reg[0]['orientation']
        if np.abs(orient_deg) < math.pi / 4:
            orient = "Vertical"
        else:
            orient = "Horizontal"

        return orient

    def getRowSortedMinMaxAvgCols(self):
        """Returns row sorted column values (row, mincol, maxcol, avgcol) ndarray
        Parameters
        ----------
        None
        Returns
        -------
        row_sorted_cols : ndarray of int
            row sorted column values (row, mincol, maxcol, avgcol) ndarray
        """

        (rows, cols) = np.nonzero(self.largestCC)
        uniq_rows = np.unique(rows)
        row_sorted_cols = np.zeros((len(uniq_rows), 4), dtype='uint16')

        for i in range(len(uniq_rows)):
            nnz_cols = np.nonzero(self.largestCC[uniq_rows[i], :])
            cur_row_tuple = tuple(map(int, (uniq_rows[i], np.min(nnz_cols), np.max(nnz_cols), np.mean(nnz_cols))))

            for j in range(4):
                row_sorted_cols[i, j] = cur_row_tuple[j]

        return row_sorted_cols

    def getColSortedMinMaxAvgRows(self):
        """Returns column sorted row values (col, minrow, maxrow, avgrow) ndarray
        Parameters
        ----------
        None
        Returns
        -------
        col_sorted_rows : ndarray of int
            column sorted row values (col, minrow, maxrow, avgrow) ndarray
        """

        (rows, cols) = np.nonzero(self.largestCC)
        uniq_cols = np.unique(cols)
        col_sorted_rows = np.zeros((len(uniq_cols), 4), dtype='uint16')

        for i in range(len(uniq_cols)):
            nnz_rows = np.nonzero(self.largestCC[:, uniq_cols[i]])
            cur_col_tuple = tuple(map(int, (uniq_cols[i], np.min(nnz_rows), np.max(nnz_rows), np.mean(nnz_rows))))

            for j in range(4):
                col_sorted_rows[i, j] = cur_col_tuple[j]

        return col_sorted_rows

    def getPolyfitCols(self, poly_deg=3):
        """Returns polynomial fit columns for each row
        Parameters
        ----------
        poly_deg : int
            degree of polynomial
        Returns
        -------
        (x, y_vals) : tuple of int
            rows and their respective polynomial fit columns
        """
        x = self.getRowSortedMinMaxAvgCols()[:, 0]
        y = self.getRowSortedMinMaxAvgCols()[:, 3]

        poly_coef = np.polyfit(x, y, poly_deg)

        y_vals = np.array(list(map(int, np.polyval(poly_coef, x))))

        return tuple(zip(x, y_vals))

    def getPolyfitRows(self, poly_deg=3):
        """Returns polynomial fit rows for each column
        Parameters
        ----------
        poly_deg : int
            degree of polynomial
        Returns
        -------
        (x, y_vals) : tuple of int
            columns and their respective polynomial fit rows
        """
        x = self.getColSortedMinMaxAvgRows()[:, 0]
        y = self.getColSortedMinMaxAvgRows()[:, 3]

        poly_coef = np.polyfit(x, y, poly_deg)

        y_vals = np.array(list(map(int, np.polyval(poly_coef, x))))

        return tuple(zip(x, y_vals))

    def getPolyfitImg(self):
        """Returns bool img with polynomial fit line
        Parameters
        ----------
        None
        Returns
        -------
        img_curve : ndarray of bool
            bool img with polynomial fit line (rows for vertical, cols for horizontal)
        """
        if self.getOrientationCC() == 'Vertical':
            row_cols_tuple = self.getPolyfitCols()
            img_curve = np.zeros((self.h, self.w), dtype='bool')
            rows = [a_tuple[0] for a_tuple in row_cols_tuple]
            cols = [a_tuple[1] for a_tuple in row_cols_tuple]
        else:
            row_cols_tuple = self.getPolyfitRows()
            img_curve = np.zeros((self.h, self.w), dtype='bool')
            cols = [a_tuple[0] for a_tuple in row_cols_tuple]
            rows = [a_tuple[1] for a_tuple in row_cols_tuple]

        img_curve[rows, cols] = True
        img_curve = self.largestCC & img_curve

        return img_curve

    def getRegGeomInfo(self):
        """Returns dictionary of geometrical information for given region
        Parameters
        ----------
        None
        Returns
        -------
        regGeomInfo : dict
            dictionary of geometrical information for given region
                "area": int
                "perim": int
                "length": int
        """

        area, perim, length = list(map(np.count_nonzero, [self.largestCC, self.boundary, self.getPolyfitImg()]))
        circ = (4*math.pi*area) / (perim**2)
        circ = builtins.round(circ, 2)

        regGeomInfo = {
            "area": area,
            "perim": perim,
            "length": length,
            "circ": circ
        }

        return regGeomInfo

    def getRegInsInfo(self, img):
        """Returns dictionary of intensity information for given region and related image
        Parameters
        ----------
        img_ : instance of ImgInfo class
        Returns
        -------
        regInsInfo : dict
            dictionary of intensity information for given region
                "rmean": float
                "gmean": float
                "bmean": float
                "grmean": float
                "rvar": float
                "gvar": float
                "bvar": float
                "rg": float (red / green)
                "gr": float (green / red)
                "rb": float (red / blue)
                "br": float (blue / red)
                "rgr": float (red / gray)
                "ggr": float (green / gray)
                "bgr": float (blue / gray)
        """

        nnz_bw_idx = np.nonzero(self.img_bw)
        
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        img_gray = 255*skimage.color.rgb2gray(img)
        
        rmean, gmean, bmean, grmean = map(np.mean, [img_r[nnz_bw_idx], img_g[nnz_bw_idx], img_b[nnz_bw_idx], img_gray[nnz_bw_idx]])
        rvar, gvar, bvar = map(np.var, [img_r[nnz_bw_idx], img_g[nnz_bw_idx], img_b[nnz_bw_idx]])

        gr, br = (gmean / rmean, bmean / rmean) if rmean else (0, 0)
        rg, bg = (rmean / gmean, bmean / gmean) if gmean else (0, 0)
        rb, gb = (rmean / bmean, gmean / bmean) if bmean else (0, 0)

        rgr, ggr, bgr = (rmean / grmean, gmean / grmean, bmean / grmean) if grmean else (0, 0, 0)

        regInsInfo = {
            "rmean": rmean, "gmean": gmean, "bmean": bmean, "grmean": grmean,
            "rvar": rvar, "gvar": gvar, "bvar": bvar,
            "rg": rg, "gr": gr,
            "rb": rb, "br": br,
            "gb": gb, "bg": bg,
            "rgr": rgr, "ggr": ggr, "bgr": bgr
        }

        regInsInfo = {k: float("%.2f" % v) for k, v in regInsInfo.items()}

        return regInsInfo