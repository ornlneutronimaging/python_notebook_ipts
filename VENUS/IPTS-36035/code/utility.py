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

from code.normalization_for_timepix import normalization_with_list_of_runs

IPTS = 36035
DEFAULT_RUN_NUMBER = None
DEFAULT_OB_RUN_NUMBER = ""

NEXUS_PATH = f"/SNS/VENUS/IPTS-{IPTS}/nexus/"


class DefaultOBRegion(Enum):
    x = 278
    y = 242
    width = 199
    height = 31


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


def define_input_full_file_name(run_number: int, ipts: int):
    sample_full_path = f"/SNS/VENUS/IPTS-{ipts}/shared/autoreduce/mcp/images/Run_{run_number}/"
    assert os.path.exists(sample_full_path), f"Path {sample_full_path} does not exist"

    # spectra file
    spectra_file_list = glob.glob(f"/SNS/VENUS/IPTS-{ipts}/shared/autoreduce/mcp/images/Run_{run_number}/*_Spectra.txt")
    if len(spectra_file_list) == 0:
        raise FileNotFoundError(f"No spectra files found in {sample_full_path}")
    spectra_file = spectra_file_list[0]
    display(HTML("Spectra file: " + spectra_file))

    # nexus file
    nexus_file = f"/SNS/VENUS/IPTS-36035/nexus/VENUS_{run_number}.nxs.h5"
    assert os.path.exists(nexus_file), f"Path {nexus_file} does not exist"

    list_tiff = glob.glob(os.path.join(sample_full_path, "*.tif"))
    list_tiff.sort()
    assert len(list_tiff) > 0, f"No tiff files found in {sample_full_path}"

    return {'sample_full_path': sample_full_path,
            'spectra_file': spectra_file,
            'nexus_file': nexus_file,
            'list_tiff': list_tiff,}


def load_tiff_files(list_tiff):
# load the first tiff file
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
    
    return lambda_axis, shutter_time

def display_integrated_signal(data, sample_run_number):
    data_integrated = np.mean(data, axis=0)
    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    im = ax.imshow(data_integrated, cmap='gray')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Integrated data from run {sample_run_number}")
    return data_integrated


class Utility:

    ob_input_file_name_dict = {}
    ob_data = {}

    def __init__(self):
        pass

    def enter_sample_run_number(self):
        label = widgets.Label("Sample run number (ex: 8737):")
        self.sample_run_number_widget = widgets.IntText(DEFAULT_RUN_NUMBER, layout=widgets.Layout(width="100px"))
        hori_layout = widgets.HBox(children=[label, self.sample_run_number_widget], layout=widgets.Layout(width="100%"))
        display(hori_layout)

    def enter_ob_run_numbers(self):
        label = widgets.Label("Open beam run numbers (ex: 8734, 8735):")
        self.ob_run_numbers_widget = widgets.Text(value = DEFAULT_OB_RUN_NUMBER, layout=widgets.Layout(width="100px"))
        hori_layout = widgets.HBox(children=[label, self.ob_run_numbers_widget], layout=widgets.Layout(width="100%"))
        display(hori_layout)

    def enter_run_numbers(self):
        self.enter_sample_run_number()
        self.enter_ob_run_numbers()

    def load_data(self):
        self.sample_run_number = self.sample_run_number_widget.value
        self.input_file_name_dict = define_input_full_file_name(self.sample_run_number, IPTS)
        self.sample_full_path = self.input_file_name_dict["sample_full_path"]
        self.spectra_file = self.input_file_name_dict["spectra_file"]
        self.nexus_file = self.input_file_name_dict["nexus_file"]
        self.list_tiff = self.input_file_name_dict["list_tiff"]

        self.data = load_tiff_files(self.list_tiff)
        self.lambda_axis, self.shutter_time = get_lambda_axis(self.nexus_file, 
                                           self.spectra_file) 
        
    # def load_sample_data(self):
    #     self.load_data()

    # def load_ob_data(self):
    #     self.ob_run_number = self.ob_run_numbers_widget.value
    #     self.ob_run_number_list = [int(_run) for _run in self.ob_run_number.split(",")]

    #     self.ob_input_file_name_dict = {}
    #     self.ob_data = {}
    #     for _run in self.ob_run_number_list:
    #         if _run == self.sample_run_number:
    #             raise ValueError(f"Open beam run number {_run} cannot be the same as sample run number {self.sample_run_number}. Please enter a different open beam run number.")

    #         self.ob_input_file_name_dict[_run] = define_input_full_file_name(_run, IPTS)
    #         self.ob_data[_run] = load_tiff_files(self.ob_input_file_name_dict[_run]["list_tiff"])

    # def load_all_data(self):
    #     self.load_sample_data()
    #     self.load_ob_data()
      
    def select_open_beam_region(self):
        self.data_integrated = np.mean(self.data, axis=0)
        def plot_roi(x, y, width, height, log_scale):
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            if log_scale:
                im = axs[0].imshow(np.log(self.data_integrated), cmap='gray')
            else:
                im = axs[0].imshow(self.data_integrated, cmap='gray')
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
            for _data in self.data:
                # calculate the mean of the region of interest
                ob_profile_region.append(np.mean(_data[y: y+height, x: x+width]))

            axs[1].plot(self.lambda_axis, ob_profile_region, 'b.')
            axs[1].set_title(f"Mean of the region of interest selected just above!")
            axs[1].set_xlabel(u"lambda (\u212B)")
            axs[1].set_ylabel("Mean counts")
            if log_scale:
                axs[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        self.ob_display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=self.data_integrated.shape[1], 
                                                            step=1, 
                                                            value=DefaultOBRegion.x.value),
                                    y=widgets.IntSlider(min=0, 
                                                        max=self.data_integrated.shape[1], 
                                                        step=1, 
                                                        value=DefaultOBRegion.y.value),
                                    width=widgets.IntSlider(min=0, 
                                                            max=self.data_integrated.shape[1], 
                                                            step=1, 
                                                            value=DefaultOBRegion.width.value),
                                    height=widgets.IntSlider(min=0, 
                                                                max=self.data_integrated.shape[1], 
                                                                step=1, 
                                                                value=DefaultOBRegion.height.value),
                                    log_scale=widgets.Checkbox(value=True, description='Log scale'),
                                                                )

        display(self.ob_display_plot_roi)

    def perform_normalization(self):
        x, y, width, height = self.ob_display_plot_roi.result

        self.normalized_data = np.zeros_like(self.data, dtype=np.float32)
        for _index, _data in enumerate(self.data):
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
                                            edgecolor='red',
                                            linewidth=2,
                                            fill=False)
            ax[0].add_patch(_rectangle1)
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title(f"Normalized data")

            sample_profile_region = []
            for _data in self.normalized_data:
                # calculate the mean of the region of interest
                sample_profile_region.append(np.mean(_data[y: y+height, x: x+width]))
            
            ax[1].plot(self.lambda_axis, sample_profile_region, 'b.')
            ax[1].set_title(f"Profile of the region selected of the normalized data")
            ax[1].set_xlabel(u"lambda (\u212B)")
            ax[1].set_ylabel("Mean counts")
            if log_scale:
                ax[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        self.sample_display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=self.data_integrated.shape[1], 
                                                            step=1, 
                                                            continuous_update=False,
                                                            value=x),
                                    y=widgets.IntSlider(min=0, 
                                                        max=self.data_integrated.shape[1], 
                                                            continuous_update=False,
                                                        step=1, 
                                                        value=y),
                                    width=widgets.IntSlider(min=0, 
                                                            max=self.data_integrated.shape[1], 
                                                            continuous_update=False,
                                                            step=1, 
                                                            value=width),
                                    height=widgets.IntSlider(min=0, 
                                                                max=self.data_integrated.shape[1], 
                                                            continuous_update=False,
                                                                step=1, 
                                                                value=height),
                                    log_scale=widgets.Checkbox(value=True, 
                                                               description='Log scale'),
           )

        display(self.sample_display_plot_roi)
   
    def perform_ob_normalization(self):
        sample_run_number = self.sample_run_number_widget.value
        self.sample_run_number = sample_run_number
        full_path_sample = [f"/SNS/VENUS/IPTS-{IPTS}/shared/autoreduce/mcp/images/Run_{sample_run_number}"]

        ob_run_numbers = self.ob_run_numbers_widget.value
        ob_run_number_list = [int(_run) for _run in ob_run_numbers.split(",")]
        full_path_ob_run_numbers = [f"/SNS/VENUS/IPTS-{IPTS}/shared/autoreduce/mcp/images/Run_{_run}" for _run in ob_run_number_list]

        output_folder = f"/SNS/VENUS/IPTS-{IPTS}/shared/processed_data/normalized_with_ob/Run_{sample_run_number}/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        nexus_path = NEXUS_PATH

        proton_charge_flag = True
        shutter_counts_flag = True
        replace_ob_zeros_by_nan_flag = True

        export_mode =  {'normalized_stack': False,
                       }
         
        all_normalized_data, self.lambda_axis, self.energy_array, self.shutter_time = normalization_with_list_of_runs(sample_run_numbers=full_path_sample,
                                                        ob_run_numbers=full_path_ob_run_numbers,
                                                        output_folder=output_folder,
                                                        nexus_path=nexus_path,
                                                        proton_charge_flag=proton_charge_flag,
                                                        shutter_counts_flag=shutter_counts_flag,
                                                        replace_ob_zeros_by_nan_flag=replace_ob_zeros_by_nan_flag,
                                                        export_mode=export_mode,
                        )

        self.normalized_data = all_normalized_data[sample_run_number]

    def interactive_profile_of_normalized_data(self):

        normalized_data_integrated = np.nanmean(self.normalized_data, axis=0)

        def plot_roi(x, y, width, height, log_scale):
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))
            if log_scale:
                im = ax[0].imshow(np.log(normalized_data_integrated), cmap='gray')
            else:
                im = ax[0].imshow(self.integrated_normalized, cmap='gray')
            _rectangle1 = patches.Rectangle((x, y),
                                            width,
                                            height,
                                            edgecolor='red',
                                            linewidth=2,
                                            fill=False)
            ax[0].add_patch(_rectangle1)
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title(f"Normalized data")

            sample_profile_region = []
            for _data in self.normalized_data:
                # calculate the mean of the region of interest
                sample_profile_region.append(np.nanmean(_data[y: y+height, x: x+width]))
            
            ax[1].plot(self.lambda_axis, sample_profile_region, 'b.')
            ax[1].set_title(f"Profile of the region selected of the normalized data")
            ax[1].set_xlabel(u"lambda (\u212B)")
            ax[1].set_ylabel("Mean counts")
            if log_scale:
                ax[1].set_yscale('log')
            plt.tight_layout()

            return x, y, width, height

        self.sample_display_plot_roi = interactive(plot_roi, 
                                    x=widgets.IntSlider(min=0, 
                                                        max=normalized_data_integrated.shape[1], 
                                                            step=1, 
                                                            continuous_update=False,
                                                            value=DefaultSampleRegion.x.value),
                                    y=widgets.IntSlider(min=0, 
                                                        max=normalized_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                        step=1, 
                                                        value=DefaultSampleRegion.y.value),
                                    width=widgets.IntSlider(min=0, 
                                                            max=normalized_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                            step=1, 
                                                            value=DefaultSampleRegion.width.value),
                                    height=widgets.IntSlider(min=0, 
                                                                max=normalized_data_integrated.shape[1], 
                                                            continuous_update=False,
                                                                step=1, 
                                                                value=DefaultSampleRegion.height.value),
                                    log_scale=widgets.Checkbox(value=True, description='Log scale'),
           )

        display(self.sample_display_plot_roi)


    def rebin(self, binning_factor: int, list_to_bin: list = None, list_lambda_axis: list = None):
        """Rebin the data based on the binning factor.

        Args:
            binning_factor (int): Factor by which to rebin the data.
            list_to_bin (list, optional): List of values to rebin. If None, uses self.profile_of_region.
        """
        end_index = 0
        y_axis_rebinned = []
        x_axis_rebinned = []
        while end_index < len(list_to_bin):
            start_index = end_index
            end_index += binning_factor
            if end_index > len(list_to_bin):
                break
            y_axis_rebinned.append(np.nanmean(list_to_bin[start_index:end_index]))
            x_axis_rebinned.append(np.nanmean(list_lambda_axis[start_index:end_index]))

        return y_axis_rebinned, x_axis_rebinned
      
    def interactive_rebin(self):
        shutter_time = self.shutter_time

        list_index_jump = np.where(np.diff(shutter_time) > 0.0001)[0]
        list_index_jump = np.insert(list_index_jump, 0, 0)  # add the first index
        list_index_jump = np.append(list_index_jump, len(shutter_time))  # add the last index
        # print(f"{list_index_jump=}")
        
        x, y, width, height = self.sample_display_plot_roi.result
        self.profile_of_region = []    
        for _data in self.normalized_data:
            # calculate the mean of the region of interest
            self.profile_of_region.append(np.nanmean(_data[y: y+height, x: x+width]))

        def rebinning(binning_factor: int):

            full_x_axis_rebinned = []
            full_y_axis_rebinned = []
            truncated_list_lambda_axis = self.lambda_axis[:]
            list_lambda_edges = []
            for _index in range(0, len(list_index_jump)-1):

                if _index == 0:
                    _start_index = list_index_jump[_index]
                else:
                    _start_index = list_index_jump[_index] + 1

                _end_index = list_index_jump[_index + 1]

                if _end_index == len(shutter_time):
                    _end_index -= 1

                if (_end_index - _start_index + 1) < len(truncated_list_lambda_axis):
                    _list_lambda_axis = truncated_list_lambda_axis[0: _end_index-_start_index]
                else:
                    _list_lambda_axis = truncated_list_lambda_axis[:]
                list_lambda_edges.append([_list_lambda_axis[0], _list_lambda_axis[-1]])

                truncated_list_lambda_axis = truncated_list_lambda_axis[_end_index-_start_index+1:]
                # print(f"\t _start_index: {_start_index}, _end_index: {_end_index}")
                y_axis_rebinned, x_axis_rebinned = self.rebin(binning_factor=binning_factor, 
                                                            list_to_bin=self.profile_of_region[_start_index:_end_index],
                                                            list_lambda_axis=_list_lambda_axis)

                full_y_axis_rebinned.extend(y_axis_rebinned)
                full_x_axis_rebinned.extend(x_axis_rebinned)

            self.full_x_axis_rebinned = full_x_axis_rebinned
            self.full_y_axis_rebinned = full_y_axis_rebinned

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.plot(full_x_axis_rebinned, full_y_axis_rebinned, '*')
            for _left, _right in list_lambda_edges:
                ax.axvspan(_left, _right, color='green', alpha=0.1)
            ax.set_title(f"Mean of the region of interest from run {self.sample_run_number}")
            ax.set_xlabel(u"Lambda (\u212b)")
            ax.set_ylabel("Mean counts")

            return binning_factor

        self.binning_factor_widget = widgets.IntSlider(description="Bin factor",
                                                       min=1, 
                                                        max=100, 
                                                        step=1, 
                                                        value=2)
        interactive_plot = interactive(rebinning, 
                                       binning_factor=self.binning_factor_widget)
        display(interactive_plot)

    def export_results(self):

        self.export_stack_of_normalized_data()
        self.export_profile_of_region()

    def export_stack_of_normalized_data(self):
        output_folder = f"/SNS/VENUS/IPTS-{IPTS}/shared/notebooks/processed_data/normalized_without_ob/Run_{self.sample_run_number}/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for _index, _image in enumerate(self.normalized_data):
            output_file = os.path.join(output_folder, f"image_{_index:04d}.tif")
            dxchange.write_tiff(_image, output_file, dtype='float32')
        print(f"Normalized data exported to {output_folder}")

    def export_profile_of_region(self):
        xaxis = self.full_x_axis_rebinned
        yaxis = self.full_y_axis_rebinned
        df = pd.DataFrame({'lambda': xaxis, 'mean_counts': yaxis})
        output_folder = f"/SNS/VENUS/IPTS-{IPTS}/shared/notebooks/processed_data/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
      
        if self.binning_factor_widget.value == 1:
            output_file = os.path.join(output_folder, f"run_{self.sample_run_number}_normalized.csv")
        else:
            output_file = os.path.join(output_folder, f"run_{self.sample_run_number}_normalized_binned_by_{self.binning_factor_widget.value}.csv")
        df.to_csv(output_file, index=False)
        print(f"ASCII file exported {output_file}")
