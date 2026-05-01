"""Mutable/connected value state-tracking primitives."""

from __future__ import annotations


class SettableValue:
    def __init__(self, default_value):
        self._value = default_value
        self._changed = True
        self._force_value = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self._changed = True

    def has_changed(self):
        return self._changed

    def reset_changed(self):
        self._changed = False


class ConnectedValue:
    def __init__(self, default_value, keep_value_on_disconnect=False):
        self._connected_node = None
        self._connected_value = None
        self._manual_value = SettableValue(default_value)
        self._keep_value_on_disconnect = keep_value_on_disconnect
        self._has_connection_changed = False

    def connect(self, node_getter):
        self._connected_value = node_getter
        self._has_connection_changed = True

    def disconnect(self):
        if self._keep_value_on_disconnect and self._connected_value is not None:
            self._manual_value.value = self._connected_value.value
        self._connected_value = None
        self._has_connection_changed = True

    @property
    def value(self):
        if self._connected_value is not None and not self._manual_value._force_value:
            return self._connected_value.value
        return self._manual_value.value

    @value.setter
    def value(self, val):
        self._manual_value.value = val

    def has_changed(self):
        return (
            (self._connected_value is not None and self._connected_value.has_changed())
            or self._manual_value.has_changed()
            or self._has_connection_changed
        )

    def reset_changed(self):
        self._manual_value.reset_changed()
        self._has_connection_changed = False
