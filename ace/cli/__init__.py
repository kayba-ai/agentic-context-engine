"""Kayba CLI."""

import click

from ace.cli.cloud import (
    upload,
    traces,
    run,
    insights,
    prompts,
    status,
    materialize,
    batch,
    setup,
    integrations,
)


@click.group()
@click.version_option(package_name="ace-framework")
def cli():
    """Kayba CLI."""
    pass


cli.add_command(upload)
cli.add_command(traces)
cli.add_command(run)
cli.add_command(insights)
cli.add_command(prompts)
cli.add_command(status)
cli.add_command(materialize)
cli.add_command(batch)
cli.add_command(setup)
cli.add_command(integrations)


def main():
    cli()
