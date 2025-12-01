from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)