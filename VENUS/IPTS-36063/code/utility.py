from enum import Enum
import glob as glob
from IPython.display import display, HTML
import os
import numpy as np
import dxchange
import h5py
import pandas as pd
from scipy.constants import h, c, electron_volt, m_n
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import display, HTML
from tqdm import tqdm

IPTS = 36063
DEFAULT_RUN_NUMBER = 8702
DEFAULT_OB_RUN_NUMBER = None   # 8703

class DefaultOBRegion(Enum):
    x = 190
    y = 195
    width = 135
    height = 135

class DefaultSampleRegion(Enum):
    x = DefaultOBRegion.x.value
    y = 300
    width = DefaultOBRegion.width.value
    height = DefaultOBRegion.height.value

class TimeUnitOptions(str, Enum):
    s = "s"
    ms = "ms"
    us = "us"
    ns = "ns"
    ps = "ps"

class DistanceUnitOptions(str, Enum):
    cm = "cm"
    nm = "nm"
    pm = "pm"
    m = "m"
    angstrom = "angstrom"

def convert_time_units(from_unit, to_unit):
    """Convert time from one unit to another unit
    based on TimeUnitOptions options

    Args:
        from_unit (TimeUnitOptions): Unit to convert from.
        to_unit (TimeUnitOptions): Unit to convert to.

    Returns:
        float: Time in the new unit.
    """

    # Conversion factors
    conversion_factors = {
        TimeUnitOptions.s: 1,
        TimeUnitOptions.ms: 1e-3,
        TimeUnitOptions.us: 1e-6,
        TimeUnitOptions.ns: 1e-9,
        TimeUnitOptions.ps: 1e-12,
    }

    return conversion_factors[from_unit] / conversion_factors[to_unit]

def convert_distance_units(from_unit, to_unit):
    """Convert distance from one unit to another unit
    based on DistanceUnitOptions options

    Args:
        from_unit (DistanceUnitOptions): Unit to convert from.
        to_unit (DistanceUnitOptions): Unit to convert to.

    Returns:
        float: distance in the new unit.
    """

    # Conversion factors
    conversion_factors = {
        DistanceUnitOptions.nm: 1e-9,
        DistanceUnitOptions.cm: 1e-2,
        DistanceUnitOptions.pm: 1e-12,
        DistanceUnitOptions.m: 1,
        DistanceUnitOptions.angstrom: 1e-10,
    }

    return conversion_factors[from_unit] / conversion_factors[to_unit]

def convert_array_from_time_to_lambda(time_array: np.ndarray, 
                                      time_unit: TimeUnitOptions,       
                                      distance_source_detector: float,
                                      distance_source_detector_unit: DistanceUnitOptions,
                                      detector_offset: float,   
                                      detector_offset_unit: DistanceUnitOptions,
                                      lambda_unit: DistanceUnitOptions) -> np.ndarray:
    """Convert an array of time values to wavelength values.

    Args:
        time_array (np.ndarray): Array of time values.
        time_unit (TimeUnitOptions): Unit of the input time.
        distance_source_detector (float): Distance from the source to the detector.
        distance_source_detector_unit (DistanceUnitOptions): Unit of the distance.
        detector_offset (float): Offset of the detector.
        detector_offset_unit (DistanceUnitOptions): Unit of the offset.
        lambda_unit (DistanceUnitOptions): Unit of the output wavelength.

    This is using the formula: lambda_m = h/(m_n * distance_source_detector_m) * (time_array_s + detector_offset_s)

    Returns:
        np.ndarray: Array of wavelength values.
    """
    time_array_s = time_array * convert_time_units(time_unit, TimeUnitOptions.s)
    detector_offset_s = detector_offset * convert_time_units(detector_offset_unit, TimeUnitOptions.s)
    distance_source_detector_m = distance_source_detector * convert_distance_units(distance_source_detector_unit, DistanceUnitOptions.m)

    h_over_mn = h / m_n
    lambda_m = h_over_mn * (time_array_s + detector_offset_s) / distance_source_detector_m

    lambda_converted = lambda_m * convert_distance_units(DistanceUnitOptions.m, lambda_unit)

    return lambda_converted

def define_input_full_file_names(run_number: int):
    sample_full_path = f"/SNS/VENUS/IPTS-{IPTS}/shared/autoreduce/mcp/images/Run_{run_number}/"
    assert os.path.exists(sample_full_path), f"Path {sample_full_path} does not exist"

    # spectra file
    spectra_file_list = glob.glob(f"/SNS/VENUS/IPTS-{IPTS}/shared/autoreduce/mcp/images/Run_{run_number}/*_Spectra.txt")
    if len(spectra_file_list) == 0:
        raise FileNotFoundError(f"No spectra files found in {sample_full_path}")
    spectra_file = spectra_file_list[0]
    #display(HTML("Spectra file: " + spectra_file))

    # nexus file
    nexus_file = f"/SNS/VENUS/IPTS-{IPTS}/nexus/VENUS_{run_number}.nxs.h5"
    assert os.path.exists(nexus_file), f"Path {nexus_file} does not exist"

    list_tiff = glob.glob(os.path.join(sample_full_path, "*.tif"))
    list_tiff.sort()
    assert len(list_tiff) > 0, f"No tiff files found in {sample_full_path}"

    return {'sample_full_path': sample_full_path,
            'spectra_file': spectra_file,
            'nexus_file': nexus_file,
            'list_tiff': list_tiff,}

def load_tiff_files(list_tiff):
    data_0 = dxchange.read_tiff(list_tiff[0])

    # set up array
    data = np.zeros((len(list_tiff), data_0.shape[0], data_0.shape[1]), dtype=data_0.dtype)
    for i, _tif in enumerate(tqdm(list_tiff)):
        data[i] = dxchange.read_tiff(_tif)

    return data

def get_lambda_axis(nexus_file: str, spectra_file: str) -> np.ndarray:

        # load from nexus
    def get_detector_offset_from_nexus(nexus_path: str) -> float:
        """get the detector offset from the nexus file"""
        with h5py.File(nexus_path, 'r') as hdf5_data:
            try:
                detector_offset_micros = hdf5_data['entry']['DASlogs']['BL10:Det:TH:DSPT1:TIDelay']['value'][0]
            except KeyError:
                detector_offset_micros = None
        return detector_offset_micros

    detector_offset_us = get_detector_offset_from_nexus(nexus_file)

    # load the spectra
    pd_spectra = pd.read_csv(spectra_file, sep=",", header=0)
    shutter_time = pd_spectra["shutter_time"].values

    lambda_axis = convert_array_from_time_to_lambda(time_array=shutter_time,
                                                time_unit=TimeUnitOptions.s,
                                                distance_source_detector=25,
                                                distance_source_detector_unit=DistanceUnitOptions.m,
                                                detector_offset=detector_offset_us,
                                                detector_offset_unit=TimeUnitOptions.us,
                                                lambda_unit=DistanceUnitOptions.angstrom)
    
    return lambda_axis

def display_integrated_signal(data, sample_run_number):
    data_integrated = np.mean(data, axis=0)
    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    im = ax.imshow(data_integrated, cmap='gray')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Integrated data from run {sample_run_number}")
    return data_integrated


class Utility:

    sample_data = None
    ob_data = None

    def __init__(self):
        pass

    def enter_input_run_numbers(self):
        label = widgets.Label("Sample run number:")
        self.sample_run_number_widget = widgets.Text(str(DEFAULT_RUN_NUMBER), 
                                                     layout=widgets.Layout(width="100px"))
        hori_layout1 = widgets.HBox(children=[label, self.sample_run_number_widget], layout=widgets.Layout(width="100%"))
        display(hori_layout1)

        # label = widgets.Label("OPTIONAL - Open beam run number:")
        # self.ob_run_number_widget = widgets.Text(str(DEFAULT_OB_RUN_NUMBER), 
        #                                             layout=widgets.Layout(width="100px"))
        # hori_layout2 = widgets.HBox(children=[label, self.ob_run_number_widget], layout=widgets.Layout(width="100%"))
        # display(hori_layout2)

        label = widgets.Label("Default sample ROI:")
        self.x_widget = widgets.IntText(DefaultSampleRegion.x.value,
                                        description='x:', 
                                        layout=widgets.Layout(width="150px"))
        self.y_widget = widgets.IntText(DefaultSampleRegion.y.value, 
                                        description='y:', 
                                        layout=widgets.Layout(width="150px"))
        self.width_widget = widgets.IntText(DefaultSampleRegion.width.value, 
                                            description='width:',
                                            layout=widgets.Layout(width="150px"))
        self.height_widget = widgets.IntText(DefaultSampleRegion.height.value, 
                                             description='height:',
                                             layout=widgets.Layout(width="150px"))
        hori_layout3 = widgets.VBox(children=[label, 
                                              self.x_widget, 
                                              self.y_widget, 
                                              self.width_widget, 
                                              self.height_widget], layout=widgets.Layout(width="100%"))
        display(hori_layout3)

    def load_and_display_data(self):
        self.load()
        self.display_integrated_signal()

    def load(self):
        self.sample_run_number = self.sample_run_number_widget.value
        self.input_file_name_dict = define_input_full_file_names(self.sample_run_number)
        self.sample_full_path = self.input_file_name_dict["sample_full_path"]
        self.spectra_file = self.input_file_name_dict["spectra_file"]
        self.nexus_file = self.input_file_name_dict["nexus_file"]
        self.list_tiff = self.input_file_name_dict["list_tiff"]
        self.sample_data = load_tiff_files(self.list_tiff)
        self.lambda_axis = get_lambda_axis(self.nexus_file, 
                                           self.spectra_file) 
        
        # self.ob_run_number = self.ob_run_number_widget.value
        # if self.ob_run_number == "None":
        #     self.ob_data = None
        #     return
        
        # if self.ob_run_number.strip() != "":
        #     self.input_ob_file_name_dict = define_input_full_file_names(self.ob_run_number)
        #     self.ob_list_tiff = self.input_ob_file_name_dict["list_tiff"]
        #     self.ob_data = load_tiff_files(self.ob_list_tiff)
        # else:
        #     self.ob_data = None
        
    def display_integrated_signal(self):
        self.sample_data_integrated = np.mean(self.sample_data, axis=0)
              
        if self.ob_data is not None:
            
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
            im = axs[0].imshow(self.sample_data_integrated, cmap='gray')
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(f"Integrated data from run {self.sample_run_number}")

            self.ob_data_integrated = np.mean(self.ob_data, axis=0)
            im2 = axs[1].imshow(self.ob_data_integrated, cmap='gray')
            plt.colorbar(im, ax=axs[1])
            axs[1].set_title(f"Integrated ob data from run {self.ob_run_number}")
            plt.tight_layout()

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 8))
            im = ax.imshow(self.sample_data_integrated, cmap='gray')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Integrated data from run {self.sample_run_number}")
      
    def sample_profile_of_region_of_interest(self):

        x = self.x_widget.value
        y = self.y_widget.value
        width = self.width_widget.value
        height = self.height_widget.value

        def plot_roi(x, y, width, height, log_flag):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
            if log_flag:
                im = axs[0].imshow(np.log(self.sample_data_integrated), cmap='gray')
            else:
                im = axs[0].imshow(self.sample_data_integrated, cmap='gray')
            
            _rectangle1 = patches.Rectangle((x, y),
                                            width,
                                            height,
                                            edgecolor='green',
                                            linewidth=2,
                                            fill=False)
            axs[0].add_patch(_rectangle1)
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(f"ROI from x:{x} y:{y} width:{width} height:{height}")

            sample_profile_region = []
            for _data in self.sample_data:
                # calculate the mean of the region of interest
                sample_profile_region.append(np.mean(_data[y: y+height, x: x+width]))
            axs[1].plot(self.lambda_axis, sample_profile_region, 'g')
            axs[1].set_title(f"Mean of the region of interest selected!")
            axs[1].set_xlabel(u"lambda (\u212B)")
            axs[1].set_ylabel("Mean counts")
            if log_flag:
                axs[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                        step=1, 
                                                        continuous_update=False,                                                      
                                                        value=x),
                                    y=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                        step=1, 
                                                        continuous_update=False,  
                                                        value=y),
                                    width=widgets.IntSlider(min=0, 
                                                            max=self.sample_data_integrated.shape[0], 
                                                            continuous_update=False, 
                                                            step=1, 
                                                            value=width),
                                    height=widgets.IntSlider(min=0, 
                                                             max=self.sample_data_integrated.shape[0], 
                                                             continuous_update=False, 
                                                             step=1, 
                                                             value=height),
                                    log_flag=widgets.Checkbox(value=False, description='Log scale'),
        )

        display(display_plot_roi)

    def export_profile_to_csv_callback(self, value):

        print("Exporting profile to CSV...")

        x = self.x_widget.value
        y = self.y_widget.value
        width = self.width_widget.value
        height = self.height_widget.value

        sample_profile_region = []
        for _data in self.sample_data:
            # calculate the mean of the region of interest
            sample_profile_region.append(np.mean(_data[y: y+height, x: x+width]))

        df = pd.DataFrame({'lambda': self.lambda_axis, 'mean_counts': sample_profile_region})
        output_file_name = f"sample_profile_run_{self.sample_run_number}_x_{x}_y_{y}_width_{width}_height_{height}.csv"
        df.to_csv(output_file_name, index=False)
        display(HTML(f"Exported profile to {output_file_name}"))
        
    def export_profile_to_csv(self):
        export_button = widgets.Button(description="Export profile to CSV")
        export_button.on_click(self.export_profile_to_csv_callback)
        display(export_button)

    def normalize_by_sample_roi(self):

        def plot_roi(x, y, width, height, log_scale):
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
            if log_scale:
                im = axs[0].imshow(np.log(self.sample_data_integrated), cmap='gray')
            else:
                im = axs[0].imshow(self.sample_data_integrated, cmap='gray')
            _rectangle1 = patches.Rectangle((x, y),
                                            width,
                                            height,
                                            edgecolor='red',
                                            linewidth=2,
                                            fill=False)
            axs[0].add_patch(_rectangle1)

            plt.colorbar(im, ax=axs[0])
            axs[0].set_title(f"ROI from x:{x} y:{y} width:{width} height:{height}")

            ob_profile_region = []
            for _data in self.sample_data:
                # calculate the mean of the region of interest
                ob_profile_region.append(np.mean(_data[y: y+height, x: x+width]))

            axs[1].plot(self.lambda_axis, ob_profile_region, 'r')
            axs[1].set_title(f"Mean of the region of interest selected")
            axs[1].set_xlabel(u"lambda (\u212B)")
            axs[1].set_ylabel("Mean counts")
            if log_scale:
                axs[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        self.ob_display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                            step=1, 
                                                            value=DefaultOBRegion.x.value),
                                    y=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                        step=1, 
                                                        value=DefaultOBRegion.y.value),
                                    width=widgets.IntSlider(min=0, 
                                                            max=self.sample_data_integrated.shape[1], 
                                                            step=1, 
                                                            value=DefaultOBRegion.width.value),
                                    height=widgets.IntSlider(min=0, 
                                                                max=self.sample_data_integrated.shape[1], 
                                                                step=1, 
                                                                value=DefaultOBRegion.height.value),
                                    log_scale=widgets.Checkbox(value=False, description='Log scale'),
                                                                )

        display(self.ob_display_plot_roi)
  
    def perform_normalization(self):
        x, y, width, height = self.ob_display_plot_roi.result

        self.normalized_data = np.zeros_like(self.sample_data, dtype=np.float32)
        for _index, _data in enumerate(self.sample_data):
            _mean_roi_counts = np.nanmean(_data[y:y+height, x:x+width])
            if _mean_roi_counts == 0:
                continue
            self.normalized_data[_index] = _data / _mean_roi_counts
            
        self.integrated_normalized = np.mean(self.normalized_data, axis=0)

        def plot_roi(x, y, width, height, log_scale):
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))
            if log_scale:
                im = ax[0].imshow(np.log(self.integrated_normalized), cmap='gray')
            else:
                im = ax[0].imshow(self.integrated_normalized, cmap='gray')
            _rectangle1 = patches.Rectangle((x, y),
                                            width,
                                            height,
                                            edgecolor='green',
                                            linewidth=2,
                                            fill=False)
            ax[0].add_patch(_rectangle1)
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title(f"Normalized data")

            sample_profile_region = []
            for _data in self.normalized_data:
                # calculate the mean of the region of interest
                sample_profile_region.append(np.mean(_data[y: y+height, x: x+width]))
            
            ax[1].plot(self.lambda_axis, sample_profile_region, 'g')
            ax[1].set_title(f"Profile of the region selected of the normalized data")
            ax[1].set_xlabel(u"lambda (\u212B)")
            ax[1].set_ylabel("Mean counts")
            if log_scale:
                ax[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        self.sample_display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                            step=1, 
                                                            continuous_update=False,
                                                            value=x),
                                    y=widgets.IntSlider(min=0, 
                                                        max=self.sample_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                        step=1, 
                                                        value=y),
                                    width=widgets.IntSlider(min=0, 
                                                            max=self.sample_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                            step=1, 
                                                            value=width),
                                    height=widgets.IntSlider(min=0, 
                                                                max=self.sample_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                                step=1, 
                                                                value=height),
                                    log_scale=widgets.Checkbox(value=False, description='Log scale'),
           )

        display(self.sample_display_plot_roi)


    # def rebin(self, binning_factor: int):

    #     x, y, width, height = self.sample_display_plot_roi.result
    #     self.profile_of_region = []    
    #     for _data in self.normalized_data:
    #         # calculate the mean of the region of interest
    #         self.profile_of_region.append(np.mean(_data[y: y+height, x: x+width]))

    #     end_index = 0
    #     rebinned_profile = []
    #     lambda_rebinned_axis = []
    #     while end_index < len(self.lambda_axis):
    #         start_index = end_index
    #         end_index += binning_factor
    #         rebinned_profile.append(np.mean(self.profile_of_region[start_index:end_index]))
    #         lambda_rebinned_axis.append(np.mean(self.lambda_axis[start_index:end_index]))

    #     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #     ax.plot(lambda_rebinned_axis, rebinned_profile, '*')
    #     ax.set_title(f"Mean of the region of interest from run {self.sample_run_number}")
    #     ax.set_xlabel(u"Lambda (\u212b)")
    #     ax.set_ylabel("Mean counts")

    # def interactive_rebin(self):
    #     def rebinning(binning_factor: int):
    #         self.rebin(binning_factor)

    #     self.binning_factor_widget = widgets.IntSlider(description="Bin factor",
    #                                                    min=1, 
    #                                                     max=100, 
    #                                                     step=1, 
    #                                                     value=2)
    #     interactive_plot = interactive(rebinning, 
    #                                    binning_factor=self.binning_factor_widget)
    #     display(interactive_plot)
