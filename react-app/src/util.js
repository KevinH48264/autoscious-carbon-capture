export const randomDarkModeColor = () => {
    // vary hue between 0 and 360 degrees
    const hue = Math.random() * 360;
    // keep saturation at 80% for rich colors
    const saturation = 80;
    // vary lightness between 30 and 70% to keep colors neither too dark nor too bright
    const lightness = 30 + Math.random() * 40;
    
    return `hsla(${hue},${saturation}%,${lightness}%, 1)`;
}

export const rectIntersectsRect = (a, b) => {
    if (a.x > b.x + b.width || a.x + a.width < b.x)
        return false;
    if (a.y > b.y + b.height || a.y + a.height < b.y)
        return false;
    return true;
}

// Sort points in counterclockwise order
export const sortPoints = (points, centerX, centerY) => {
    return points.sort((a, b) => {
    return Math.atan2(a[1] - centerY, a[0] - centerX) - Math.atan2(b[1] - centerY, b[0] - centerX);
    });
}