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

// creating clusters
export function getLeafClusters(clusterData) {
    let clusterNodes = [];
  
    function recurse(cluster, parentsMap) {
      cluster.voronoiPoints = [];

      const {cluster_id, layer, centroid_x, centroid_y, polygonPoints, content} = cluster;

      if (typeof cluster.content[0] === 'object' && cluster.content[0] !== null) {
        parentsMap = new Map(parentsMap); // Create a new map at each layer so it doesn't get overridden in sibling branches
        parentsMap.set(layer, cluster_id); // Map the current layer to the current cluster_id
        cluster.content.forEach((child) => recurse(child, parentsMap));
      } else {
        parentsMap.set(layer, cluster_id); // Map the current layer to the current cluster_id
        const parents = {};
        for(let [key, value] of parentsMap.entries()) {
          parents[key] = value;
        }
        clusterNodes.push({cluster_id, layer, centroid_x, centroid_y, content, polygonPoints, parents});
      }
    }
  
    clusterData.forEach((cluster) => recurse(cluster, new Map()));
  
    return clusterNodes;
}
  

  // Assigning Voronoi points to clusters
  export function flattenClusters(clusterData, voronoi, leafClusters) {
    let clusterNodes = [];

    function recurse(cluster) {
      const {cluster_id, layer, centroid_x, centroid_y, content} = cluster;
      const leafClusterIndex = leafClusters.findIndex(leafCluster => leafCluster.cluster_id === cluster_id);

    //   console.log("cluster", cluster, "leafClusterIndex", leafClusterIndex)

      // if it's a leaf cluster with Voronoi points
      if (leafClusterIndex !== -1) {
        // console.log(" 2 voronoi.cellPolygon(0)", voronoi.cellPolygon(0))
        const cellPolygon = voronoi.cellPolygon(leafClusterIndex);

        if (cellPolygon) {
          cluster.voronoiPoints = cellPolygon;
        } else {
          cluster.voronoiPoints = [];
        }
        
        // console.log("cluster.cluster_id", cluster.cluster_id, "ALLPOINTS", cellPolygon)
      }

      // otherwise it's a parent cluster
      else {
        cluster.content.forEach(recurse);

        // If it's a parent cluster, gather Voronoi points from child clusters
        let allPoints = cluster.content.flatMap(child => child.voronoiPoints);
        // console.log("cluster.cluster_id", cluster.cluster_id, "ALLPOINTS", allPoints)

        if (allPoints[0] === []) {
          cluster.voronoiPoints = []
        } else {
          let centerX = allPoints.reduce((sum, point) => sum + point[0], 0) / allPoints.length;
          let centerY = allPoints.reduce((sum, point) => sum + point[1], 0) / allPoints.length;

          cluster.voronoiPoints = sortPoints(allPoints, centerX, centerY);
        }
      } 

      clusterNodes.push({cluster_id, layer, centroid_x, centroid_y, content, voronoiPoints: cluster.voronoiPoints});
      // console.log("done pushing!")
    }

    // console.log("clusterData", clusterData)
    clusterData.forEach(recurse);
    console.log("DONE RECURSING!", "clusterNodes", clusterNodes)

    return clusterNodes;
  }