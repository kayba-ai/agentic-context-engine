"""Kayba CLI."""

import click

from ace.cli.cloud import upload, insights, prompts, status, materialize, batch


@click.group()
@click.version_option(package_name="ace-framework")
def cli():
    """Kayba CLI."""
    pass


cli.add_command(upload)
cli.add_command(insights)
cli.add_command(prompts)
cli.add_command(status)
cli.add_command(materialize)
cli.add_command(batch)


def main():
    cli()
