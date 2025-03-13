import os

os.environ['SCIPY_ARRAY_API'] = "1"

from nicegui import app, ui, events
from io import StringIO
import pandas as pd

from file_upload import upload_file
from run import run_analysis


import re

app.native.window_args['resizable'] = False
app.native.start_args['debug'] = True
app.native.settings['ALLOW_DOWNLOADS'] = True

ui.label('Data Processing Kitchen Sink').classes('text-2xl font-bold')

design_matrix = None
quantification_matrix = None


params = dict()

with ui.tabs() as tabs:
    home = ui.tab('home', label='Home', icon='home')
    database_upload = ui.tab('Upload')
    normalization = ui.tab('Normalize')
    batch_correction = ui.tab('Correct')
    protein_quantification = ui.tab('Quantify')
    imputation = ui.tab('Impute')
    differential_abundance = ui.tab('Compare')
    xml = ui.tab('Explain')
    feature_selection = ui.tab('Select')

normalize = None
correct = None
quantify = None
impute = None
compare = None
explain = None
select = None

#with ui.grid(rows=1, columns=3).style('min-width: 1000px; max-width: 1000px; min-height: 500px; max-height: 500px;'):
with ui.splitter() as splitter:

    with splitter.before:
        with ui.column().style('min-width: 100px; max-width: 100px;'):

            normalize = ui.checkbox("Perform Normalization", value=False)
            correct = ui.checkbox("Perform Batch Correction", value=False)
            quantify = ui.checkbox("Perform Quantification", value=False)
            impute = ui.checkbox("Perform Imputation", value=False)
            compare = ui.checkbox("Perform Differential Comparison", value=False)
            explain = ui.checkbox("Perform ML Explaination", value=False)
            select = ui.checkbox("Perform Feature Selection", value=False)

    with splitter.after:

        with ui.splitter() as splitter2:

            with splitter2.before:

                with ui.timeline(side="right").style('min-width: 200px; max-width: 200px;'):
                    with ui.timeline_entry(
                        "Normalize",
                        title="Normalize",
                    ).bind_visibility_from(normalize, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize")).props(f"height=10px")

                    with ui.timeline_entry(
                        "Correct",
                        title="Correct",
                    ).bind_visibility_from(correct, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))

                    with ui.timeline_entry(
                        "Quantify",
                        title="Quantify",
                    ).bind_visibility_from(quantify, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))

                    with ui.timeline_entry(
                        "Impute",
                        title="Impute",
                    ).bind_visibility_from(impute, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))

                    with ui.timeline_entry(
                        "Compare",
                        title="Compare",
                    ).bind_visibility_from(compare, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))

                    with ui.timeline_entry(
                        "Explain",
                        title="Explain",
                    ).bind_visibility_from(explain, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))

                    with ui.timeline_entry(
                        "Select",
                        title="Select",
                    ).bind_visibility_from(select, 'value'):
                        ui.button("Configure", on_click=lambda: tabs.set_value("Normalize"))
            with splitter2.after:


                with ui.tab_panels(tabs, value=home).style('min-width: 700px; max-width: 700px;') as panels:

                    with ui.tab_panel(home):

                        ui.label("Home")


                    with ui.tab_panel(database_upload):

                        ui.label('Upload Design Matrix')
                        design_matrix = upload_file(ui)

                        ui.label('Upload Quantification Matrix')
                        quantification_matrix = upload_file(ui)


                    with ui.tab_panel(normalization):

                        params['normalize'] = dict()
                        ui.label('Normalization')

                        ui.label("Normalization method:")

                        with ui.row():
                            normalization_method = ui.radio(
                                [
                                    "tic",
                                    "median",
                                    "mean",
                                    "log2"
                                ], value="mean"
                            ).props('inline')

                            params['normalize']['method'] = normalization_method.value

                        log2_transform = ui.checkbox("Log2 Transform?")

                        params['normalize']['log2'] = log2_transform.value

                        use_rt_window = ui.checkbox("Use Retention time sliding window?", value=True)

                        params['normalize']['use_rt_window'] = use_rt_window.value

                        with ui.row().bind_visibility_from(use_rt_window, 'value'):

                            minimum_data_points = ui.number(
                                label='Minimum Data Points',
                                value=100,
                                precision=0
                            )
                            minimum_data_points.sanitize()

                            params['normalize']['minimum_data_points'] = minimum_data_points.value

                            stride = ui.number(
                                label='Stride',
                                value=5,
                                precision=0
                            )
                            stride.sanitize()

                            params['normalize']['stride'] = stride.value

                            use_overlapping_windows = ui.checkbox("Use Overlapping Windows?", value=True)

                            params['normalize']['use_overlapping_windows'] = use_overlapping_windows.value

                    with ui.tab_panel(batch_correction):

                        params['correct'] = dict()

                        ui.markdown("# Batch Correction")
                        ui.markdown("A batch must be included in the design matrix for this section.")

                        ui.label("Batch correction method:")

                        with ui.row():
                            batch_correction_method = ui.radio(
                                [
                                    "mean",
                                ], value="mean"
                            ).props('inline')

                            params['correct']['method'] = batch_correction_method.value

                            reference_batch = ui.number(
                                label='Reference Batch',
                                value=1,
                                precision=0
                            )
                            reference_batch.sanitize()

                            params['correct']['reference_batch'] = reference_batch.value

                    with ui.tab_panel(protein_quantification):

                        params['quantify'] = dict()

                        ui.markdown("# Quantification")
                        ui.markdown("Quantification of proteins (or similar) from peptides.")

                        ui.label("Quantification level")

                        quantification_level = ui.radio(
                            [
                                "protein",
                                "peptide",
                            ], value="protein"
                        ).props('inline')

                        params['quantify']['level'] = quantification_level.value

                        ui.label("Quantification Method")

                        with ui.row():
                            quantification_method = ui.radio(
                                [
                                    "top_n",
                                    "maxlfq"
                                ], value="maxlfq"
                            ).props('inline')

                            params['quantify']['method'] = quantification_method.value

                        with ui.row().bind_visibility_from(quantification_method, 'value', value="top_n"):

                            top_n = ui.number(
                                label='Top N',
                                value=3,
                                precision=0
                            )
                            top_n.sanitize()

                            params['quantify']['top_n'] = top_n.value

                            summarization_method = ui.radio(
                                [
                                    "sum",
                                    "mean",
                                    "median"
                                ], value="sum"
                            ).props('inline')

                            params['quantify']['summarization_method'] = summarization_method.value

                        with ui.row().bind_visibility_from(quantification_method, 'value', value="maxlfq"):

                            threads = ui.number(
                                label='Threads',
                                value=1,
                                precision=0
                            )
                            threads.sanitize()

                            params['quantify']['threads'] = threads.value

                            min_subgroups = ui.number(
                                label='Minimum Subgroups',
                                value=1,
                                precision=0
                            )
                            min_subgroups.sanitize()

                            params['quantify']['minimum_subgroups'] = min_subgroups.value

                            top_n = ui.number(
                                label='Top N',
                                value=3,
                                precision=0
                            )
                            top_n.sanitize()

                            params['quantify']['top_n'] = top_n.value

                    with ui.tab_panel(imputation):

                        params['impute'] = dict()

                        ui.markdown("# Imputation")
                        ui.markdown("Impute missing values of quantified analytes.")

                        imputation_method = ui.radio(
                            [
                                "Uniform Percentile",
                                "Uniform Range",
                                "Constant",
                                "Neighborhood"
                            ], value="Uniform Percentile"
                        )

                        params['impute']['method'] = imputation_method.value

                        with ui.row().bind_visibility_from(imputation_method, 'value', value="Uniform Percentile"):

                            percentile = ui.number(
                                label='Percentile',
                                value=0.1,
                                precision=3
                            )
                            percentile.sanitize()

                            params['impute']['percentile'] = percentile.value

                        with ui.row().bind_visibility_from(imputation_method, 'value', value="Uniform Range"):

                            minvalue = ui.number(
                                label='Min Value',
                                value=0,
                                precision=3
                            )
                            minvalue.sanitize()

                            params['impute']['minvalue'] = minvalue.value

                            maxvalue = ui.number(
                                label='Max Value',
                                value=1,
                                precision=3
                            )
                            maxvalue.sanitize()

                            params['impute']['maxvalue'] = maxvalue.value

                        with ui.row().bind_visibility_from(imputation_method, 'value', value="Constant"):

                            constant = ui.number(
                                label="Constant Value",
                                value=0,
                                precision=3
                            )
                            constant.sanitize()

                            params['impute']['constant'] = constant.value

                        with ui.row().bind_visibility_from(imputation_method, 'value', value="Neighborhood"):

                            n_neighbors = ui.number(
                                label="Number of Neighbors",
                                value=5,
                                precision=0
                            )
                            n_neighbors.sanitize()

                            params['impute']['n_neighbors'] = n_neighbors.value

                            weights = ui.radio(
                                [
                                    "distance",
                                    "uniform",
                                ], value="distance"
                            ).props('inline')

                            params['impute']['weights'] = weights.value

                    with ui.tab_panel(differential_abundance):

                        params['compare'] = dict()

                        ui.markdown("# Statistical Comparison")
                        ui.markdown("Statisitical comparisons between experimental groups indicated in the design matrix.")

                        ui.label("Comparison Method")
                        comparison_method = ui.radio(
                            [
                                "ttest",
                                "linregress",
                                "anova",
                                "ttest_paired"
                            ], value="linregress"
                        )

                        params['compare']['method'] = comparison_method.value

                        ui.label("Comparisons")
                        comparisons = ui.input(
                            value='Comparisons',
                        ).props('clearable')

                        params['compare']['comparisons'] = [tuple(x.split(',')) for x in re.findall("\((.*?)\)", comparisons.value)]

                        print(params['compare']['comparisons'])

                        min_samples_per_group = ui.number(
                            label='Percentile',
                            value=0.1,
                            precision=3
                        )
                        min_samples_per_group.sanitize()

                        params['compare']['min_samples_per_group'] = min_samples_per_group.value

                        ui.label("Multiple Testing Correction Method")
                        multiple_testing_correction_method = ui.select(
                            [
                                "fdr_tsbh",
                                "fdr_bh",
                                "fdr_by",
                                "fdr_tsbky"
                            ],
                            value="fdr_tsbh"
                        )

                        params['compare']['multiple_testing_correction_method'] = multiple_testing_correction_method.value

                    with ui.tab_panel(xml):

                        params['explain'] = dict()

                        ui.markdown("# Explain Predictions")
                        ui.markdown("Identify important proteins using explainable machine learning.")

                        ui.label("Comparisons")
                        comparisons = ui.input(
                            value='Comparisons',
                        ).props('clearable')

                        params['compare']['comparisons'] = [tuple(x.split(',')) for x in re.findall("\((.*?)\)", comparisons.value)]

                        n_iterations = ui.number(
                            label='Number of Iterations',
                            value=100,
                            precision=0
                        )
                        n_iterations.sanitize()

                        params['explain']['n_iterations'] = n_iterations.value

                        downsample_background = ui.checkbox(
                            "Downsample Background", value=True
                        )

                        params['explain']['downsample_background'] = downsample_background.value

                        feature_column = ui.input(
                            value='Feature Column',
                        )

                        params['explain']['feature_column'] = feature_column.value

                        fillna = ui.checkbox("Fill NA's")

                        params['explain']['fillna'] = fillna.value

                    with ui.tab_panel(feature_selection):
                        ui.label('Normalization')


def run_():
    ui.label("Running")
    print(params)
    run_analysis(params)


ui.button('Run Analysis', on_click=run_)

ui.run(native=True, fullscreen=False, window_size=(1500, 800))
