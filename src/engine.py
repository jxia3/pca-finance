import dataclasses
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Any

from src.data import DailyReturns
from src.models.model import Model
from src.modules.module import Module

class Collection(Enum):
    LIST = 0
    ARRAY = 1

@dataclass
class ModelEntry:
    name: str
    model: Model
    collection: Collection = Collection.LIST

class Engine:
    daily_returns: DailyReturns
    models: list[ModelEntry]
    modules: list[Module]
    executed: bool

    def __init__(self, daily_returns: DailyReturns, models: list[ModelEntry], modules: list[Module]) -> None:
        assert len(daily_returns.dates) > 0 and len(daily_returns.dates) == len(daily_returns.returns)
        assert len(models) == len(set(entry.name for entry in models))
        self.daily_returns = daily_returns
        self.models = models
        self.modules = modules
        self.executed = False

    def run(self) -> dict[str, Any]:
        assert not self.executed
        self.executed = True
        collections = {}

        for d in range(len(self.daily_returns.dates)):
            timestamp = self.daily_returns.dates[d]
            returns = self.daily_returns.returns[:d + 1]
            model_outputs = {}

            for entry in self.models:
                output = entry.model.update(timestamp, returns, model_outputs)
                self._collect_output(collections, entry, d, output)
                model_outputs[entry.name] = self._create_view(collections, entry, d)
            for module in self.modules:
                module.update(timestamp, returns, model_outputs)

        return collections

    def _collect_output(
        self,
        collections: dict[str, Any],
        entry: ModelEntry,
        index: int,
        output: Any,
    ) -> None:
        if entry.collection != Collection.ARRAY:
            if entry.name not in collections:
                collections[entry.name] = []
            collections[entry.name].append(output)
            return

        if isinstance(output, np.ndarray):
            if entry.name not in collections:
                collections[entry.name] = np.zeros(
                    shape=(len(self.daily_returns.dates), *output.shape),
                    dtype=output.dtype,
                )
            assert isinstance(collections[entry.name], np.ndarray)
            assert output.shape == collections[entry.name].shape[1:]
            collections[entry.name][index] = output
            return

        assert dataclasses.is_dataclass(output) and not isinstance(output, type)
        for field in dataclasses.fields(output):
            assert isinstance(getattr(output, field.name), np.ndarray)
        if entry.name not in collections:
            collections[entry.name] = type(output)(**{
                field.name: np.zeros(
                    shape=(len(self.daily_returns.dates), *getattr(output, field.name).shape),
                    dtype=getattr(output, field.name).dtype,
                )
                for field in dataclasses.fields(output)
            })

        assert isinstance(collections[entry.name], type(output))
        for field in dataclasses.fields(output):
            collection = getattr(collections[entry.name], field.name)
            assert isinstance(collection, np.ndarray)
            assert getattr(output, field.name).shape == collection.shape[1:]
            collection[index] = getattr(output, field.name)

    def _create_view(self, collections: dict[str, Any], entry: ModelEntry, index: int) -> Any:
        assert entry.name in collections
        value = collections[entry.name]
        if entry.collection != Collection.ARRAY:
            return value

        if isinstance(value, np.ndarray):
            return value[:index + 1]
        assert dataclasses.is_dataclass(value) and not isinstance(value, type)
        for field in dataclasses.fields(value):
            assert isinstance(getattr(value, field.name), np.ndarray)

        return type(value)(**{
            field.name: getattr(value, field.name)[:index + 1]
            for field in dataclasses.fields(value)
        })