import { forceSimulation, forceManyBody, forceCenter, forceCollide, forceLink } from 'd3-force';

function normalizeDensity(nodes, target_density = 0.0007, x0=0, y0=0) {
    const max_norm = Math.max(...nodes.map(node => Math.sqrt(node.x * node.x + node.y * node.y)));
    const area = Math.PI * max_norm * max_norm;
    const target_area = nodes.length / target_density;
    const norm_scale = Math.sqrt(target_area / area);
    for (let node of nodes) {
        node.x *= norm_scale;
        node.y *= norm_scale;

        node.x += x0;
        node.y += y0;
    }

    return max_norm * norm_scale;
}

export function computeLayout(paperNodes, edgesData, leafClusters, centroidNodes) {
    // Create dummy 'center' nodes and add them to paperNodes first
    centroidNodes.forEach(cluster => {
        // Create dummy 'center' node
        let centerNode = {
          paperId: "center_" + cluster.cluster_id, // Give it a unique id
          citationCount: 0, // Dummy value, not used for anything
          x: cluster.x,
          y: cluster.y,
          classification_id: cluster.classification_id,
          };
        paperNodes.push(centerNode);
    });

  let simulation = forceSimulation()
    .nodes(paperNodes) // Set nodes
    .force("charge", forceManyBody().strength(-50))
    .force("link", forceLink().id(d => d.paperId)) // This is how we access the id of each node
    .force("center", forceCenter(0, 0)) // Initial center, this will be updated for each leaf cluster
    .force("collision", forceCollide().radius(function(d) {
        // TODO: this is usually determined by the radius, update radius if not citationCount!
      return Math.sqrt(Math.sqrt(d.citationCount));
    }))
    .force("repel", forceManyBody()
        .strength((node) => node.paperId.startsWith("center_") ? -50 : 0)
        .distanceMax(200))
    .stop();

  let links = [];

    // Create links from edgesData (they should only be in the same cluster now)
    edgesData.forEach(edge => {
        // Find the clusters for source and target
        const sourceCluster = leafClusters.find(cluster => cluster.papers.includes(edge.source));
        const targetCluster = leafClusters.find(cluster => cluster.papers.includes(edge.target));

        // If both nodes are in the same cluster, add the link
        // if (sourceCluster && targetCluster && sourceCluster.cluster_id === targetCluster.cluster_id) {
          // if (sourceCluster && targetCluster) {
            links.push({
                source: edge.source,
                target: edge.target,
                strength: (edge.weight) ** 2, // higher value means stronger force ie nodes are pulled closer together
                distance: 50 // this specifies the desired distance for all nodes
            });
          // }
        // }
    });
  

  // Create dummy 'center' nodes and links to them for each leafCluster
  let idCounter = edgesData.length;
  leafClusters.forEach(cluster => {
    // Link all nodes in the cluster to the 'center' node
    if (cluster.papers.length !== 0) {
      cluster.papers.forEach(paperId => {
        links.push({
          source: paperId,
          target: "center_" + cluster.id,
          weight: 10 // Large weight to keep them close
        });

        // Add to edgesData for visualization
        const newEdge = {
          id: idCounter++,
          source: paperId,
          target: "center_" + cluster.id,
          weight: Math.sqrt(10) // Large weight to keep them close
        };
        edgesData.push(newEdge); // Add the new edge to edgesData
      });

      // Link clusters to parent layer - 1 above it if it exists
      if (cluster.layer - 1 >= 0) {
        const parentId = cluster.parents[cluster.layer - 1]
        
        links.push({
          source: "center_" + parentId,
          target: "center_" + cluster.id,
          weight: 100 // Large weight to keep them close
        });

        // Add to edgesData for visualization
        const newEdge = {
          id: idCounter++,
          source: "center_" + parentId,
          target: "center_" + cluster.id,
          weight: Math.sqrt(10) // Large weight to keep them close
        };
        edgesData.push(newEdge); // Add the new edge to edgesData
      }

      // Update the center force to the centroid of the current cluster
      // simulation.force("center", forceCenter(cluster.centroid_x, cluster.centroid_y));
    }    
  });


  // Set links
  simulation.force("link").links(links);

   // Add a repulsive force between centroid nodes
//    simulation.force("repel", forceManyBody().strength(-2000).distanceMax(200));

  // Manually iterate the simulation
  let normalizedRadius = 0;
  for (var i = 0; i < 300; ++i) {
    simulation.tick();
    normalizedRadius = normalizeDensity(paperNodes, 0.0007);
  }

  // Remove the dummy 'center' nodes before returning
  let centerNodes = paperNodes.filter(node => node.paperId.startsWith("center_"));
  paperNodes = paperNodes.filter(node => !node.paperId.startsWith("center_"));

  // Now paperNodes have their 'final' position computed by d3 force layout
  return {paperNodes, centerNodes, normalizedRadius, edgesData};
}
