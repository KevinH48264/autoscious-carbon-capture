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
}

export function computeLayout(paperNodes, edgesData, leafClusters) {
    // Create dummy 'center' nodes and add them to paperNodes first
    leafClusters.forEach(cluster => {
        // Create dummy 'center' node
        let centerNode = {
        paperId: "center_" + cluster.cluster_id, // Give it a unique id
        citationCount: 0 // Dummy value, not used for anything
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

    // Create links from edgesData
    edgesData.forEach(edge => {
        // Find the clusters for source and target
        const sourceCluster = leafClusters.find(cluster => cluster.content.includes(edge.source));
        const targetCluster = leafClusters.find(cluster => cluster.content.includes(edge.target));

        // If both nodes are in the same cluster, add the link
        if (sourceCluster && targetCluster && sourceCluster.cluster_id === targetCluster.cluster_id) {
            links.push({
                source: edge.source,
                target: edge.target,
                strength: edge.weight ** 2,
                distance: 10
            });
        }
    });
  

  // Create dummy 'center' nodes and links to them for each leafCluster
  leafClusters.forEach(cluster => {
    // Link all nodes in the cluster to the 'center' node
    cluster.content.forEach(paperId => {
      links.push({
        source: paperId,
        target: "center_" + cluster.cluster_id,
        weight: 10 // Large weight to keep them close
      });
    });

    // Update the center force to the centroid of the current cluster
    // console.log("centroid for cluster ", cluster.cluster_id, cluster.centroid_x, cluster.centroid_y)
    simulation.force("center", forceCenter(cluster.centroid_x, cluster.centroid_y));
  });


  // Set links
  simulation.force("link").links(links);

   // Add a repulsive force between centroid nodes
//    simulation.force("repel", forceManyBody().strength(-2000).distanceMax(200));

  // Manually iterate the simulation
  for (var i = 0; i < 300; ++i) {
    simulation.tick();
    normalizeDensity(paperNodes, 0.0007);
  }

  // Remove the dummy 'center' nodes before returning
  let centerNodes = paperNodes.filter(node => node.paperId.startsWith("center_"));
  paperNodes = paperNodes.filter(node => !node.paperId.startsWith("center_"));

  // Now paperNodes have their 'final' position computed by d3 force layout
  return {paperNodes, centerNodes};
}
