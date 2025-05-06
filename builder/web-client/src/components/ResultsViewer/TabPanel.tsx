import React, { memo } from 'react';
import { Box } from '@mui/material';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

export const TabPanel: React.FC<TabPanelProps> = memo(({ children, value, index }) => {
  const isSelected = value === index;
  
  return (
    <div
      role="tabpanel"
      hidden={!isSelected}
      id={`results-tabpanel-${index}`}
      aria-labelledby={`results-tab-${index}`}
    >
      {isSelected && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
});

TabPanel.displayName = 'TabPanel'; 