from nicegui import app, ui, events
from io import StringIO
import pandas as pd

from file_upload import upload_file

app.native.window_args['resizable'] = True
app.native.start_args['debug'] = True
app.native.settings['ALLOW_DOWNLOADS'] = True

ui.label('Data Processing Kitchen Sink').classes('text-2xl font-bold')

design_matrix = None
quantification_matrix = None

with ui.splitter(value=30).classes('w-full') as splitter:
    with splitter.before:
        with ui.tabs().props('vertical') as tabs:
            home = ui.tab('home', label='Home', icon='home')
            database_upload = ui.tab('file_upload')
            normalization = ui.tab('Normalization')
            batch_correction = ui.tab('Batch Correction')
            protein_quantification = ui.tab('Protein Quantification')
            imputation = ui.tab('Imputation')
            differential_abundance = ui.tab('Differential Abundance')
            xml = ui.tab('XML')
            feature_selection = ui.tab('Feature Selection')

    with splitter.after:
        with ui.tab_panels(tabs, value=home).props('vertical').classes('w-full h-full'):

            with ui.tab_panel(home):

                ui.label("Home")

            with ui.tab_panel(database_upload):

                ui.label('Upload Design Matrix')
                design_matrix = upload_file(ui)

                ui.label('Upload Quantification Matrix')
                quantification_matrix = upload_file(ui)


            with ui.tab_panel(normalization):
                ui.label('Normalization')

            with ui.tab_panel(batch_correction):
                ui.label('Normalization')

            with ui.tab_panel(protein_quantification):
                ui.label('Normalization')

            with ui.tab_panel(imputation):
                ui.label('Normalization')

            with ui.tab_panel(differential_abundance):
                ui.label('Normalization')

            with ui.tab_panel(xml):
                ui.label('Normalization')

            with ui.tab_panel(feature_selection):
                ui.label('Normalization')




ui.run(native=True, fullscreen=False)
