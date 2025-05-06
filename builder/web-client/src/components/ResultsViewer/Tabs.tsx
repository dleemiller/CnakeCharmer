import React, { memo } from 'react';
import { Box, Tabs as MuiTabs, Tab } from '@mui/material';

interface TabsProps {
  value: number;
  onChange: (event: React.SyntheticEvent, newValue: number) => void;
}

export const Tabs: React.FC<TabsProps> = memo(({ value, onChange }) => {
  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
      <MuiTabs value={value} onChange={onChange} aria-label="results tabs">
        <Tab label="Summary" id="tab-0" aria-controls="tabpanel-0" />
        <Tab label="Detailed Analysis" id="tab-1" aria-controls="tabpanel-1" />
      </MuiTabs>
    </Box>
  );
});

Tabs.displayName = 'ResultsTabs'; 