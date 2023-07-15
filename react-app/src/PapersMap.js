import React, { useEffect, useRef } from 'react';
import * as PIXI from 'pixi.js';
import { Viewport } from 'pixi-viewport';
import { Delaunay } from 'd3-delaunay';
import { randomDarkModeColor, rectIntersectsRect, sortPoints, getLeafClusters, flattenClusters, diamondPolygonColorSequence, multilineText, labelBounds, getColorAtZoomLevel, traverseCluster, calculateClusterCentroids, getVoronoiNeighbors, circleColorDiamondSequence } from './util';
import { computeLayout } from './layout';

const ResearchPaperPlot = ({ papersData, edgesData, clusterData }) => {
  const pixiContainer = useRef();
  PIXI.BitmapFont.from("TitleFont", { fill: 0x000000 }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });
  PIXI.BitmapFont.from("TopicFont", { fill: 0x000000 }, { chars: PIXI.BitmapFont.ASCII.concat(['∀']) });

  useEffect(() => {
    const logging = false;
    console.log("papersData", papersData, "edgesData", edgesData, "clusterData", clusterData)

    // Compute force-directed layout of PaperNodes
    let paperNodes = papersData
      // .filter(({paperId, title, abstract}) => paperId != null && title != null && abstract != null)
      // .slice(0, 30)
      .map(({title, x, y, citationCount, paperId, abstract, classification_ids}) => ({title, x: x, y: y, citationCount, paperId, abstract, classification_ids}))
    let leafClusters = flattenClusters(clusterData);
    let centroidNodes = []
    calculateClusterCentroids(leafClusters, paperNodes, centroidNodes)
    console.log("leaf clusters", leafClusters) // expect 58 leaf clusters bc there are 58 categories
    console.log("centroidNodes", centroidNodes) // expect 58 centroid ndoes for each cluster

    const layout = computeLayout(paperNodes, edgesData, leafClusters, centroidNodes, edgesData);
    paperNodes = layout.paperNodes;
    const edges = layout.edgesData;
    const centerNodes = layout.centerNodes; // clusterNode!
    const normalizedRadius = layout.normalizedRadius; // this should be used to determine the zoom, currently 230
    const zoomScale = normalizedRadius / 150;


    const app = new PIXI.Application({
      width: window.innerWidth,
      height: window.innerHeight - 1000,
      resolution: window.devicePixelRatio || 1,
      autoDensity: true,
      backgroundColor: 0xD6EFFF,
      resizeTo: window,
    });

    if (pixiContainer.current && pixiContainer.current.childNodes.length > 0) {
      pixiContainer.current.replaceChild(app.view, pixiContainer.current.childNodes[0]);
    } else {
      pixiContainer.current.appendChild(app.view);
    }

    // For now, make the world square and even for better circle parsing
    const viewport = new Viewport({
      screenWidth: window.innerWidth,
      screenHeight: window.innerHeight,
      worldWidth: window.innerWidth,
      worldHeight: window.innerHeight,
      ticker: app.ticker,
      events: app.renderer.events,
      stopPropagation: true,
    });
    viewport.sortableChildren = true;
    viewport.drag().pinch().wheel().decelerate()
    // .clampZoom({ minWidth: 50, maxHeight: viewport.worldHeight / zoomScale, maxWidth: viewport.worldWidth / zoomScale})
    // .setZoom(zoomScale)
    .moveCenter(0, 0)
    // viewport.clamp({direction: 'all'})
    app.stage.addChild(viewport);

    // Creates a map from cluster_id to main_topic
    const clusterMap = new Map();
    clusterData.forEach(cluster => traverseCluster(cluster, clusterMap));

    const min_scale = Math.min(...paperNodes.map((node) => Math.sqrt(node.citationCount)));
    const max_scale = Math.max(...paperNodes.map((node) => Math.sqrt(node.citationCount)));
    
    // Calculating voronoi from true centroids of leaf clusters
    const extendFactor = 100 // hardcoding for circle design
    const delaunay = Delaunay.from(centroidNodes.map((node) => [node.x, node.y]));
    const minX = Math.min(...paperNodes.map((paper) => paper.x));
    const maxX = Math.max(...paperNodes.map((paper) => paper.x));
    const minY = Math.min(...paperNodes.map((paper) => paper.y));
    const maxY = Math.max(...paperNodes.map((paper) => paper.y));
    const voronoi = delaunay.voronoi([minX - extendFactor, minY - extendFactor, maxX + extendFactor, maxY + extendFactor]);
    // scale the data to fit within the worldWidth and worldHeight
    const scaleX = (d) => ((d - minX) / (maxX - minX)) * viewport.worldWidth;
    const scaleY = (d) => ((d - minY) / (maxY - minY)) * viewport.worldHeight;

    // Adding a circle mask
    let farthestDistance = 0;
    paperNodes.forEach(node => {
      let distance = Math.sqrt(Math.pow(node.x, 2) + Math.pow(node.y, 2));
      if (distance > farthestDistance) {
        farthestDistance = distance;
      }
    });
    // let circleMask = new PIXI.Graphics();
    // circleMask.beginFill(0x000000); // You can fill with any color
    // circleMask.drawCircle(0, 0, farthestDistance + 10);
    // circleMask.endFill();
    // const polygonContainer = new PIXI.Container();
    // viewport.addChild(polygonContainer);
    // polygonContainer.mask = circleMask;
    // viewport.addChild(circleMask);

    // Ensuring parentIds extend to the farthest zoom
    const zoomLayers = 20
    leafClusters.forEach((node) => {
        let lastParentId = node.cluster_id; // fallback to node's cluster_id
        for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
            if (node.parents[zoomLevel] !== undefined) {
                lastParentId = node.parents[zoomLevel];
            } else {
                node.parents[zoomLevel] = lastParentId;
            }
        }
    });

    // Creates a map from parentId to nodes with that parentId
    const clusterGroups = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
        leafClusters.forEach((node, i) => {
            const parentId = node.parents[zoomLevel];
            const key = [parentId, zoomLevel];
            if (!clusterGroups.has(key)) {
                clusterGroups.set(key, []);
            }
            // Get the indices of Voronoi neighbors for the current leaf cluster
            const neighborIndices = getVoronoiNeighbors(voronoi, i);
            // Check if any of the neighbors belong to the same parent cluster
            const neighborNodes = neighborIndices
                .map(index => leafClusters[index])
                .filter(neighbor => neighbor && neighbor.parents[zoomLevel] === parentId);
            // Add the current node and its same-parent neighbors to the group
            clusterGroups.get(key).push(node, ...neighborNodes);
        });
    }

    // Calculates centroid for each cluster
    const clusterCentroids = new Map();
    for (let [[clusterId, zoomLevel], nodes] of clusterGroups) {
        let sumX = 0, sumY = 0;
        for (let node of nodes) {
            sumX += node.centroid_x;
            sumY += node.centroid_y;
        }
        const key = [clusterId, zoomLevel].toString();
        clusterCentroids.set(key, {x: sumX / nodes.length, y: sumY / nodes.length, layer: zoomLevel});
    }

    // Hardcoding (zoomLayers) a parent cluster mapping for voronois
    let parentColorMap = new Map();
    for (let zoomLevel = 0; zoomLevel < zoomLayers; ++zoomLevel) {
      // Setting parent cluster colors by cluster_id
      leafClusters.forEach(node => {
        let parentId = node.parents[zoomLevel];
        if (parentId) {
          if (!parentColorMap.has(parentId)) {
              parentColorMap.set(parentId, diamondPolygonColorSequence[parentId % diamondPolygonColorSequence.length]);
          }
        }
      });
    }
    parentColorMap.set(0, 0xffffff); // hardcoding because 0 isn't covered for some reason

    // Sort paperNodes by citationCount to prioritize showing higher citationCount papers
    paperNodes.sort((a, b) => b.citationCount - a.citationCount);

    console.log("clusterCentroids", clusterCentroids)

    // assuming `nodes` is your array of node objects and `edgesData` is your array of edges
    let paperIdToNode = {};
    paperNodes.forEach(node => {
        paperIdToNode[node.paperId] = node;
    });
    console.log("paperIdToNode", paperIdToNode)

    const taxonomy = {
      "Carbon Capture and Storage (CCS) Technologies": {
        "Membrane-Based Technologies": ["df275f0b3dca8552250d569db1bf18de8463c9e9", "deff0e3386caac160ecf968815a2938a542e34f0", "2366c402cf265deb94ed2e6e80375b3d9bf7004c", "cfa569019717301c352aabec292a8f4d3a261553"],
        "Mineralization": ["76a20bec8e313ced0faf8010e6ecbe3965a9e305"],
        "Metal-Organic Frameworks (MOFs)": ["5b190c555003e154bb9038844899014425236273", "9308b1d5019588e96d23b401d2f9df01df00d393", "7fd50091d43cf35198cb0533a7dd3595cec7f1a7", "1aeced2070a2e23757d95aac9ebddf2e0a9d0e09", "e02f812df3cd677ec931b0ce47e10640f16968fa", "9496966a27c7a9176ec0a8ff6a04781caa42b7bb"],
        "Covalent Organic Frameworks": ["7a2197144b3a1567e814ea35fd73bd85a698681f"],
        "Direct Air Capture": ["522986777b3989c34f0ea37a35e1c4f4fb3d65c8"],
        "Pyrogenic": ["24bd758dbdb9ce0ea8e913b56a8224c0389c83fe"]
      },
      "Industry Applications": {
        "Chemical Industry": ["e24d6d882c78389a9c356b87ac46e065087549cf"],
        "Flue Gas Capture": ["19b604c4e00c43cfd916603d20b2532ccb2896ea"],
        "Natural Gas & Hydrogen": ["620999ecca4608a89bd69cbe97ab752ce1e62161"],
        "Electricity Grids": ["f15a2d21309f429fbec688b8aee5c7bdb66b8f40"]
      },
      "Socio-Economic Analysis": {
        "Life Cycle Assessment": ["154b9fd69570539e90f51e1b19db944713d3bfdc", "0ffa1b15908277b77ad7382c7b28f2e63226aabc", "12a27df31fd069ab0db9abcbd7776de7a6fee0eb"],
        "Economic Analyses": ["62519425cbe4ca9578c88b21c57553b3fd5783bb"],
        "Policy Analysis": ["6901beee29fd0c97005ff92a7aefc5c70d19aab5"],
        "Infrastructure": ["be026181508eca3fc0ce54bb3571f4fdbc014a8b"]
      },
      "Carbon Capture from Biological Sources": {
        "Natural Forests": ["1f61df1af871412b52c5fc38902854fbede1a703"],
        "Microalgae Biorefinery": ["57eec80dc233eb48b42b6ef4299acbc565a32f72"],
        "Biorefinery": ["fa71f477a4f42d0dd2b5e3b1d36d35108654ecb7"]
      },
      "Literature Reviews and Updates": {
        "General Analysis": ["e3e850fb87909bd91aab7ecb8260417ccfcb383b"],
        "Literature Review Update": ["e1e15aa932ac61efa8bb0c5cc99cfe6521458861", "3c87b0e4d3d93128fb6d425da491a7fea528b587"]
      }
    };

    // Create and add all circles and text to the viewport
    const drawNodes = (nodes, vis_cluster_centroids, viewport) => {
      let zoomLevel = Math.max(-1, Math.round(((viewport.scaled) - 1) * 5))
      zoomLevel = 9
      // let zoomLevel = Math.max(-1, Math.round(((viewport.scaled / zoomScale) - 1) * 5))
      let originalZoomLevel = zoomLevel;

      // Font size
      const bounds = viewport.getVisibleBounds();
      let min_font_size = bounds.width < bounds.height
          ? bounds.width / (28.3)
          : bounds.height / (50);
      let max_font_size = min_font_size * 1.7;

      const debug_factor = 2
      min_font_size -= debug_factor;
      max_font_size -= debug_factor


      const addedTextBounds = new Set();

      // Preview Zoom: Adding cluster polygons on the preview layer to the viewport
      // leafClusters.forEach((node, i) => {
      //   // If no parentId, then take the highest parent key
      //   let parentId = node.parents[zoomLevel + 1];
      //   if (!parentId) {
      //     parentId = node.parents[Math.max(...Object.keys(node.parents).map(Number))];
      //   }
      //   let fillColor = parentColorMap.get(parentId);

      //   const region = voronoi.cellPolygon(i);
      //   const polygon = new PIXI.Graphics();
      //   polygon.zIndex = 50;

      //   polygon.beginFill(fillColor, 0.75);
      //   polygon.drawPolygon(region.map(([x, y]) => new PIXI.Point(x, y)));
      //   polygon.endFill();

      //   node.region = polygon;
      //   polygonContainer.addChild(polygon);
      // });

      // change zoomLayers to maxZoomLayer
      // Current Zoom: Adding the cluster text to viewport, and any in the next 3 layers
      // for (let i = 0; i < 3; ++i) {
      //   vis_cluster_centroids.forEach((centroid, key) => {
      //     let [clusterId] = key.split(',').map(Number); 
      //     if (centroid.layer === zoomLevel + i) {
      //       let topCategory = "Unknown";
      //       if(clusterMap.has(clusterId)){
      //           topCategory = clusterMap.get(clusterId);
      //       }
      //       topCategory = topCategory.slice(1, -1);

      //       // Create new text
      //       let current_centroid_font_size = max_font_size / (1.1 ** (zoomLevel - originalZoomLevel));
      //       let current_centroid_text = multilineText(topCategory, 15);

      //       // Check for font size bounds
      //       // if (current_centroid_font_size < min_font_size) {
      //       //   return
      //       // }

      //       // Not allowing more than 20 labels
      //       if (addedTextBounds.length > 20) {
      //         return
      //       }

      //       // Check for overlaps with existing labels
      //       let current_zoom_text_bound = labelBounds(current_centroid_font_size, centroid.x, centroid.y, 15, current_centroid_text);

      //       for (let bound of addedTextBounds) {
      //         if (rectIntersectsRect(current_zoom_text_bound, bound)) {
      //           return
      //         }
      //       }
      //       addedTextBounds.add(current_zoom_text_bound);
            
      //       if (!centroid.current_zoom_text) {
      //           centroid.current_zoom_text = new PIXI.BitmapText(current_centroid_text, {
      //               fontFamily: 'Arial',
      //               fontSize: current_centroid_font_size,
      //               fontName: "TopicFont",
      //               align: 'left',
      //               visible: true,
      //           });
      //           centroid.current_zoom_text.zIndex = 100;

      //           // Position the text at the centroid of the cluster
      //           centroid.current_zoom_text.position.set(centroid.x, centroid.y);
      //           centroid.current_zoom_text.anchor.set(0.5, 0);

      //           // Add the text to the viewport
      //           viewport.addChild(centroid.current_zoom_text);
      //       } else {
      //           centroid.current_zoom_text.fontSize = current_centroid_font_size;
      //           centroid.current_zoom_text.visible = true;
      //       }
      //     } 
      //   });
      // }     

      console.log("leafClusters", leafClusters)
      // Adding paper nodes circles to viewport by leaf cluster
      // leafClusters.forEach(cluster => {
      //   let contentSet = new Set(cluster.papers);
      //   let leafClusterNodes = nodes.filter(node => contentSet.has(node.paperId));
        // leafClusterNodes.forEach(node => {
          paperNodes.forEach(node => {
          // Handling Node text, draw labels
          const lambda = (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
          // const circleHeight = 5 + (min_font_size / 3) * lambda;
          const circleHeight = 15
          if(!node.circle) {
              node.circle = new PIXI.Graphics();
              node.circle.zIndex = 55;
              // node.circle.beginFill(0xb9f2ff);
              // node.circle.beginFill(circleColorDiamondSequence[cluster.cluster_id % circleColorDiamondSequence.length]);
              node.circle.beginFill(0x000000);
              node.circle.drawCircle(scaleX(node.x), scaleY(node.y), circleHeight);
              node.circleHeight = circleHeight;
              node.circle.endFill();
              viewport.addChild(node.circle);
          } else {
              node.circle.visible = true;
          }

          if(!node.topic_text) {
              node.topic_text = new PIXI.BitmapText(multilineText(node.classification_ids.join(" "), 40), {
                fontFamily: 'Arial',
                fontSize: 20,
                fontName: "TitleFont",
                fill: 0xffffff,
                align: 'left',
                visible: true,
              });
              node.topic_text.zIndex = 60;
              node.topic_text.anchor.set(0.5, -0.5);
              node.topic_text.position.set(scaleX(node.x) + node.circleHeight, scaleY(node.y) - node.circleHeight - 30);
              viewport.addChild(node.topic_text);
          } else {
              node.topic_text.fontSize = 15;
              node.topic_text.visible = true; // make it visible if it already exists
          }
        });
      // })

      // Adding paper text labels to viewport by leaf cluster
      leafClusters.forEach(cluster => {
        let contentSet = new Set(cluster.papers);
        let leafClusterNodes = nodes.filter(node => contentSet.has(node.paperId));

        leafClusterNodes.forEach((node, i) => {
          // paperNodes.forEach((node, i) => {
          // console.log("node", i, node.x, node.y)
          // Handling Node text, draw labels
          const lambda = (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
          // const fontSize = (min_font_size + (max_font_size - min_font_size) * lambda / 3);
          const fontSize = 15
          let multilineTitle = multilineText(node.title, 30)
          // Using the node.title, find the topic and subtopic in the taxonomy
          // Using the node.paperId, find the topic and subtopic in the taxonomy
          for (let topic in taxonomy) {
            for (let subtopic in taxonomy[topic]) {
              if (taxonomy[topic][subtopic].includes(node.paperId)) {
                node.topic = topic;
                node.subtopic = subtopic;
                break;
              }
            }
          }
          
          // if (!node.topic || !node.subtopic) {
          //   // If the paperId is not in the taxonomy, assign a default value or handle it differently
          //   node.topic = "Unknown";
          //   node.subtopic = "Unknown";
          //   console.log("UNKNOWN", node.paperId, node.title, node.topic, node.subtopic)
          // }
          // Not allowing more than 10 paper labels / a lot of words
          // if (addedTextBounds.length > 10) {
          //   return
          // }

          // Check for overlaps with existing labels
          let current_zoom_text_bound = labelBounds(fontSize, scaleX(node.x), scaleY(node.y), 30, multilineTitle);
          for (let bound of addedTextBounds) {
            if (rectIntersectsRect(current_zoom_text_bound, bound)) {
              return
            }
          }
          addedTextBounds.add(current_zoom_text_bound);

          if(!node.text) {
              node.text = new PIXI.BitmapText(multilineText(
                  // node.topic + " ; " + node.subtopic + " ; " + 
                  node.title, 30
                ), {
                  fontFamily: 'Arial',
                  fontSize: fontSize,
                  fontName: "TitleFont",
                  fill: 0xffffff,
                  align: 'left',
                  visible: true,
                });
              node.text.zIndex = 60;
              node.text.anchor.set(0.5, 0);
              node.text.position.set(scaleX(node.x) + node.circleHeight, scaleY(node.y) + node.circleHeight + 1);
              viewport.addChild(node.text);
          } else {
              node.text.fontSize = fontSize;
              node.text.visible = true; // make it visible if it already exists
          }
        });
      })

      // Visualizing centroid nodes from force directed simulation
      layout.centerNodes.forEach((node, i) => {  
        // Handling Node text, draw labels
        const debug_factor = 1
        const lambda = debug_factor
        // const lambda = debug_factor * (Math.sqrt(node.citationCount) - min_scale) / (max_scale - min_scale);
        const fontSize = min_font_size + (max_font_size - min_font_size) * lambda;
        const circleHeight = 10;

        if(!node.circle) {
            node.circle = new PIXI.Graphics();
            node.circle.beginFill(0xFFA500);
            node.circle.drawCircle(scaleX(node.x), scaleY(node.y), circleHeight);
            node.circle.endFill();
            node.circle.zIndex = 55;
            viewport.addChild(node.circle);
        } else {
            node.circle.visible = true; // make it visible if it already exists
        }
      })

      // Add edges between nodes
      edges.forEach(edge => {
        const sourceNode = paperIdToNode[edge.source];

        if (edge.target.includes("center_")) {
          // it's a centroid

          const targetNode = layout.centerNodes.find(node => node.paperId === edge.target); // this can be optimized
          console.log("targetNode", targetNode, layout.centerNodes)

          // Create a new graphics object for the edge if it doesn't exist
          if (!edge.edge_graphics) {
              edge.edge_graphics = new PIXI.Graphics();
              edge.edge_graphics.zIndex = 50; // set this below node's zIndex to ensure nodes are drawn on top
              viewport.addChild(edge.edge_graphics);
          } 
          // Performance optimization: ?
          // else {
          //   edge.edge_graphics.visible = true;
          // }
      
          // Draw the line
          edge.edge_graphics.clear(); // remove any existing line
          edge.edge_graphics.visible = true;
          edge.edge_graphics.lineStyle(2, 0x808080, edge.weight); // set the line style (you can customize this)
          edge.edge_graphics.moveTo(scaleX(sourceNode.x), scaleY(sourceNode.y)); // move to the source node's position
          edge.edge_graphics.lineTo(scaleX(targetNode.x), scaleY(targetNode.y)); // draw a line to the target node's position
          viewport.addChild(edge.edge_graphics)
        } else {
          const targetNode = paperIdToNode[edge.target];
      
          // Create a new graphics object for the edge if it doesn't exist
          if (!edge.edge_graphics) {
              edge.edge_graphics = new PIXI.Graphics();
              edge.edge_graphics.zIndex = 50; // set this below node's zIndex to ensure nodes are drawn on top
              viewport.addChild(edge.edge_graphics);
          } 
          // Performance optimization: ?
          // else {
          //   edge.edge_graphics.visible = true;
          // }
      
          // Draw the line
          edge.edge_graphics.clear(); // remove any existing line
          edge.edge_graphics.visible = true;
          edge.edge_graphics.lineStyle(2, 0xFF0000, edge.weight * 5 ); // set the line style (you can customize this)
          edge.edge_graphics.moveTo(scaleX(sourceNode.x), scaleY(sourceNode.y)); // move to the source node's position
          edge.edge_graphics.lineTo(scaleX(targetNode.x), scaleY(targetNode.y)); // draw a line to the target node's position
          viewport.addChild(edge.edge_graphics)
        }
      });
    }

    // Update visibility of circles and text based on the current field of view and zoom level
    let totalUpdateTime = 0
    let numUpdates = 0
    let prev_viewport_bounds = PIXI.Rectangle.EMPTY;
    let count = 0
    const updateNodes = () => {
      // Start the timer
      const t0 = performance.now();
      if (!paperNodes || count > 0) return;

      // get the current field of view
      const viewport_bounds = viewport.getVisibleBounds();
			viewport_bounds.pad(viewport_bounds.width * 0.2);

      // Update visibility of nodes and labels
      if (
        prev_viewport_bounds.x !== viewport_bounds.x ||
        prev_viewport_bounds.y !== viewport_bounds.y ||
        prev_viewport_bounds.width !== viewport_bounds.width ||
        prev_viewport_bounds.height !== viewport_bounds.height
      ) {
        // reset all nodes and labels graphics not in viewport (resetting text globally was messing up the preventing text overlap and deteching text.visible)
        // leafClusters.forEach((node, i) => {
        //   if (node.region) { polygonContainer.removeChild(node.region); };
        // })
        // clusterCentroids.forEach((centroid, key) => {
        //   if (centroid.current_zoom_text) { centroid.current_zoom_text.visible = false; };
        // })
        paperNodes.forEach((node, i) => {
          if (node.circle) { node.circle.visible = false; };
          if (node.text) { node.text.visible = false; };
          if (node.graphics) { node.graphics.visible = false; };
        })
        let vis_nodes = paperNodes.filter((node) =>
          viewport_bounds.contains(node.x, node.y)
        )
        let vis_cluster_centroids = new Map();
        clusterCentroids.forEach((centroid, key) => {
          if (viewport_bounds.contains(centroid.x, centroid.y)) {
            vis_cluster_centroids.set(key, centroid);
          }
        });

        // Take the top visible nodes
        vis_nodes.sort((a, b) => {
          return b.citationCount - a.citationCount;
        });
        // vis_nodes = vis_nodes.slice(0, 25);

        prev_viewport_bounds = viewport_bounds.clone(); // clone the rectangle to avoid reference issues
        // drawNodes(vis_nodes, vis_cluster_centroids, viewport);
        drawNodes(paperNodes, clusterCentroids, viewport);

        count += 1
      }

      // Performance debugger: Stop the timer and print the time taken, 15 ms is the threshold for smooth animation (60 fps)
      if (logging) {
        const t1 = performance.now();
        const updateTime = t1 - t0;

        if (numUpdates % 60 === 0) {
          totalUpdateTime = 0
          numUpdates = 0
        }

        numUpdates += 1
        totalUpdateTime += updateTime

        console.log("Update time (ms): " + Math.round((t1 - t0)), "avg: ", Math.round(totalUpdateTime / numUpdates));
      }
    };

    // Update nodes based on ticker
    updateNodes() // for debugging
    // app.ticker.add(updateNodes)

  }, [papersData, edgesData, clusterData]);

  return <div className="pixiContainer" style={{ display: "flex" }} ref={pixiContainer} />;
};

export default ResearchPaperPlot;