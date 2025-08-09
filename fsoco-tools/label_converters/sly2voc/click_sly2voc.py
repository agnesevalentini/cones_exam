import click

from .sly2voc import main


@click.command()
@click.argument("sly_project_folder", type=str)
@click.argument("output_folder", type=str)
@click.option("--remove_watermark", is_flag=True, default=False)
@click.option("--merge", is_flag=True, default=False)
def sly2voc(sly_project_folder, output_folder, remove_watermark, merge):
    """
    Supervisely  => Pascal VOC format

    https://docs.supervise.ly/ann_format/

    \b
    Input:
    project_name
    ├── meta.json
    └── dataset_name
        ├── ann
        │   ├── img_x.json
        │   ├── img_y.json
        │   └── img_z.json
        └── img
            ├── img_x.jpeg
            ├── img_y.jpeg
            └── img_z.jpeg


    \b
    Output (if --merge, otherwise separate output for each dataset_name):
    output_folder
    ├──JPEGImages
    │  ├── img_x.jpeg
    │  ├── img_y.jpeg
    │  └── img_z.jpeg
    ├── Annotations
    │  ├── img_x.xml
    │  ├── img_y.xml
    │  └── img_z.xml
    └── ImageSets
        └── Layout
            └── <empty>

    """
    click.echo("[LOG] Running Supervisely to Pascal VOC label converter")
    main(sly_project_folder, output_folder, remove_watermark, merge)


if __name__ == "__main__":
    click.echo(
        "[LOG] This sub-module is not meant to be run as a stand-alone script. Please refer to\n $ fsoco --help"
    )
