from nicegui import events
from io import StringIO
import pandas as pd


def upload_file(local_ui):

    uploaded_file = None

    def handle_upload(e: events.UploadEventArguments):
        # Decode the uploaded file content
        content = e.content.read().decode('utf-8')
        # Use StringIO to simulate a file object
        file = StringIO(content)
        # Read the TSV content into a Pandas DataFrame
        uploaded_file = pd.read_csv(file, sep='\t')

        local_ui.table.from_pandas(
            uploaded_file,
            row_key="sample",
            pagination=10
        )


    local_ui.upload(on_upload=handle_upload).props('accept=.tsv')

    return uploaded_file