import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertForSequenceClassification, RobertaForSequenceClassification,
    OPTForSequenceClassification, get_linear_schedule_with_warmup, EarlyStoppingCallback
)
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import os
from dataclasses import dataclass
from tqdm import tqdm
import copy
import pandas as pd
import gc
import wandb
import argparse
from collections import Counter