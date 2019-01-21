#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by youshaox on 1/1/19
"""
function:

"""
import sys
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help="The person to greet.")
def hello(count, name):
    """
    Simple program that greets NAME for a total of COUNT times.
    :param count:
    :param name:
    :return:
    """
    for x in range(count):
        click.echo("hello %s" % name)

if __name__ == "__main__":
    hello()