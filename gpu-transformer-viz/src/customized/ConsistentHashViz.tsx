import { useState, useEffect, useRef, useCallback } from 'react';
import { Hash, Plus, Minus, RotateCcw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';

export const metadata = {
  name: "Consistent Hashing",
  icon: "Hash"
};

interface Server {
  id: string;
  vnodeId: string;
  angle: number;
  color: string;
}

interface Key {
  id: string;
  angle: number;
}

function hash(str: string): number {
  let h = 5381;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) + h) + str.charCodeAt(i);
  }
  return Math.abs(h) % 360;
}

function isTooClose(angle: number, existingAngles: number[], minSeparation: number): boolean {
  for (const existing of existingAngles) {
    let diff = Math.abs(angle - existing);
    diff = Math.min(diff, 360 - diff);
    if (diff < minSeparation) {
      return true;
    }
  }
  return false;
}

function findGoodAngle(baseStr: string, existingAngles: number[], minSeparation: number): number {
  let attempt = 0;
  let angle: number;
  do {
    angle = hash(baseStr + '_' + attempt);
    attempt++;
  } while (isTooClose(angle, existingAngles, minSeparation) && attempt < 100);
  return angle;
}

function generateColor(index: number): string {
  const hue = (index * 137.5) % 360;
  return `hsl(${hue}, 70%, 60%)`;
}

export default function ConsistentHashViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [servers, setServers] = useState<Server[]>([]);
  const [keys, setKeys] = useState<Key[]>([]);
  const [serverColors, setServerColors] = useState<string[]>([]);
  const [virtualNodes, setVirtualNodes] = useState(3);
  const [nextServerId, setNextServerId] = useState(0);
  const [nextKeyId, setNextKeyId] = useState(0);
  const [keyAssignments, setKeyAssignments] = useState<Map<string, string>>(new Map());
  const [lastMovedKeys, setLastMovedKeys] = useState(0);
  const [lastAction, setLastAction] = useState('');

  const findServerForKey = useCallback((keyAngle: number, serverList: Server[]): Server | null => {
    if (serverList.length === 0) return null;
    for (const server of serverList) {
      if (server.angle >= keyAngle) {
        return server;
      }
    }
    return serverList[0];
  }, []);

  const calculateKeyMovements = useCallback((action: string, serverList: Server[], keyList: Key[], oldAssignments: Map<string, string>) => {
    let movedCount = 0;
    const newAssignments = new Map<string, string>();

    keyList.forEach(key => {
      const server = findServerForKey(key.angle, serverList);
      if (server) {
        const serverId = server.id;
        newAssignments.set(key.id, serverId);
        const oldServerId = oldAssignments.get(key.id);
        if (oldServerId && oldServerId !== serverId) {
          movedCount++;
        }
      }
    });

    setKeyAssignments(newAssignments);
    setLastMovedKeys(movedCount);
    setLastAction(action);
    return newAssignments;
  }, [findServerForKey]);

  const addServer = useCallback(() => {
    const serverId = `S${nextServerId}`;
    const colorIndex = Math.floor(servers.length / virtualNodes);
    const color = generateColor(colorIndex);

    const existingAngles = servers.map(s => s.angle);
    const newServers: Server[] = [];

    for (let i = 0; i < virtualNodes; i++) {
      const vnodeId = `${serverId}-v${i}`;
      const angle = findGoodAngle(vnodeId, [...existingAngles, ...newServers.map(s => s.angle)], 20);
      newServers.push({ id: serverId, vnodeId, angle, color });
    }

    const updatedServers = [...servers, ...newServers].sort((a, b) => a.angle - b.angle);
    setServers(updatedServers);
    setServerColors([...serverColors, color]);
    setNextServerId(nextServerId + 1);

    const newAssignments = calculateKeyMovements(`Added ${serverId}`, updatedServers, keys, keyAssignments);

    // Update assignments for existing keys
    keys.forEach(key => {
      const server = findServerForKey(key.angle, updatedServers);
      if (server) {
        newAssignments.set(key.id, server.id);
      }
    });
    setKeyAssignments(newAssignments);
  }, [servers, serverColors, virtualNodes, nextServerId, keys, keyAssignments, calculateKeyMovements, findServerForKey]);

  const removeServer = useCallback(() => {
    if (servers.length <= virtualNodes) return;

    const uniqueServers = [...new Set(servers.map(s => s.id))];
    const serverToRemove = uniqueServers[uniqueServers.length - 1];

    const updatedServers = servers.filter(s => s.id !== serverToRemove);
    setServers(updatedServers);
    setServerColors(serverColors.slice(0, -1));

    calculateKeyMovements(`Removed ${serverToRemove}`, updatedServers, keys, keyAssignments);
  }, [servers, serverColors, virtualNodes, keys, keyAssignments, calculateKeyMovements]);

  const addKey = useCallback(() => {
    const keyId = `key${nextKeyId}`;
    const existingAngles = keys.map(k => k.angle);
    const angle = findGoodAngle(keyId, existingAngles, 8);
    const newKey = { id: keyId, angle };
    const updatedKeys = [...keys, newKey];
    setKeys(updatedKeys);
    setNextKeyId(nextKeyId + 1);

    if (servers.length > 0) {
      const server = findServerForKey(angle, servers);
      if (server) {
        setKeyAssignments(prev => new Map(prev).set(keyId, server.id));
      }
    }
  }, [keys, nextKeyId, servers, findServerForKey]);

  const reset = useCallback(() => {
    setServers([]);
    setKeys([]);
    setServerColors([]);
    setNextServerId(0);
    setNextKeyId(0);
    setKeyAssignments(new Map());
    setLastMovedKeys(0);
    setLastAction('');
  }, []);

  // Initialize on mount
  useEffect(() => {
    if (servers.length === 0 && nextServerId === 0) {
      // Add 3 initial servers
      let currentServers: Server[] = [];
      let currentColors: string[] = [];
      let serverId = 0;

      for (let s = 0; s < 3; s++) {
        const sId = `S${serverId}`;
        const color = generateColor(s);
        currentColors.push(color);
        const existingAngles = currentServers.map(srv => srv.angle);

        for (let i = 0; i < virtualNodes; i++) {
          const vnodeId = `${sId}-v${i}`;
          const angle = findGoodAngle(vnodeId, existingAngles, 20);
          currentServers.push({ id: sId, vnodeId, angle, color });
          existingAngles.push(angle);
        }
        serverId++;
      }

      currentServers.sort((a, b) => a.angle - b.angle);
      setServers(currentServers);
      setServerColors(currentColors);
      setNextServerId(serverId);

      // Add 10 initial keys
      let currentKeys: Key[] = [];
      const newAssignments = new Map<string, string>();
      for (let k = 0; k < 10; k++) {
        const keyId = `key${k}`;
        const existingKeyAngles = currentKeys.map(key => key.angle);
        const angle = findGoodAngle(keyId, existingKeyAngles, 8);
        currentKeys.push({ id: keyId, angle });

        const server = findServerForKey(angle, currentServers);
        if (server) {
          newAssignments.set(keyId, server.id);
        }
      }
      setKeys(currentKeys);
      setNextKeyId(10);
      setKeyAssignments(newAssignments);
    }
  }, []);

  // Handle virtual nodes change
  const updateVirtualNodes = useCallback((value: number) => {
    const currentKeyCount = keys.length;
    const currentServerCount = new Set(servers.map(s => s.id)).size;

    let currentServers: Server[] = [];
    let currentColors: string[] = [];
    let serverId = 0;

    for (let s = 0; s < currentServerCount; s++) {
      const sId = `S${serverId}`;
      const color = generateColor(s);
      currentColors.push(color);
      const existingAngles = currentServers.map(srv => srv.angle);

      for (let i = 0; i < value; i++) {
        const vnodeId = `${sId}-v${i}`;
        const angle = findGoodAngle(vnodeId, existingAngles, 20);
        currentServers.push({ id: sId, vnodeId, angle, color });
        existingAngles.push(angle);
      }
      serverId++;
    }

    currentServers.sort((a, b) => a.angle - b.angle);

    let currentKeys: Key[] = [];
    const newAssignments = new Map<string, string>();
    for (let k = 0; k < currentKeyCount; k++) {
      const keyId = `key${k}`;
      const existingKeyAngles = currentKeys.map(key => key.angle);
      const angle = findGoodAngle(keyId, existingKeyAngles, 8);
      currentKeys.push({ id: keyId, angle });

      const server = findServerForKey(angle, currentServers);
      if (server) {
        newAssignments.set(keyId, server.id);
      }
    }

    setVirtualNodes(value);
    setServers(currentServers);
    setServerColors(currentColors);
    setKeys(currentKeys);
    setKeyAssignments(newAssignments);
    setLastAction('');
    setLastMovedKeys(0);
  }, [keys.length, servers, findServerForKey]);

  // Draw on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 280;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the ring
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw server ranges (colored arcs)
    for (let i = 0; i < servers.length; i++) {
      const server = servers[i];
      const nextServer = servers[(i + 1) % servers.length];

      const startAngle = (server.angle - 90) * Math.PI / 180;
      let endAngle = (nextServer.angle - 90) * Math.PI / 180;

      if (nextServer.angle < server.angle) {
        endAngle += 2 * Math.PI;
      }

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, endAngle);
      ctx.strokeStyle = server.color;
      ctx.lineWidth = 20;
      ctx.globalAlpha = 0.3;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw server nodes
    servers.forEach(server => {
      const angle = (server.angle - 90) * Math.PI / 180;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);

      // Outer glow
      ctx.beginPath();
      ctx.arc(x, y, 18, 0, 2 * Math.PI);
      ctx.fillStyle = server.color;
      ctx.globalAlpha = 0.2;
      ctx.fill();
      ctx.globalAlpha = 1;

      // Main node
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, 2 * Math.PI);
      ctx.fillStyle = server.color;
      ctx.fill();
      ctx.strokeStyle = '#1e40af';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Label
      const labelX = x;
      const labelY = y - 28;
      const labelText = server.vnodeId;
      ctx.font = 'bold 12px sans-serif';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = 'white';
      ctx.fillRect(labelX - textWidth / 2 - 4, labelY - 12, textWidth + 8, 16);

      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'center';
      ctx.fillText(labelText, labelX, labelY);
    });

    // Draw keys
    keys.forEach(key => {
      const server = findServerForKey(key.angle, servers);
      const angle = (key.angle - 90) * Math.PI / 180;
      const x = centerX + (radius - 60) * Math.cos(angle);
      const y = centerY + (radius - 60) * Math.sin(angle);

      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = server?.color || '#10b981';
      ctx.fill();
      ctx.strokeStyle = '#059669';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw line to assigned server
      if (server) {
        const serverAngle = (server.angle - 90) * Math.PI / 180;
        const serverX = centerX + radius * Math.cos(serverAngle);
        const serverY = centerY + radius * Math.sin(serverAngle);

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(serverX, serverY);
        ctx.strokeStyle = server.color;
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.15;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // Draw angle markers
    ctx.fillStyle = '#9ca3af';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    [0, 90, 180, 270].forEach(angle => {
      const rad = (angle - 90) * Math.PI / 180;
      const x = centerX + (radius + 35) * Math.cos(rad);
      const y = centerY + (radius + 35) * Math.sin(rad);
      ctx.fillText(`${angle}°`, x, y + 5);
    });

    // Clockwise label
    ctx.fillStyle = '#6b7280';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('← Clockwise lookup →', centerX, centerY - radius - 40);

    // Center label
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText('Hash Ring', centerX, centerY - 10);
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.fillText('(0° - 360°)', centerX, centerY + 10);
  }, [servers, keys, findServerForKey]);

  const uniqueServerCount = new Set(servers.map(s => s.id)).size;

  return (
    <div className="max-w-5xl mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            <Hash className="text-blue-600" size={32} />
            Consistent Hashing Visualization
          </CardTitle>
          <p className="text-gray-500 text-sm mt-1">See how keys redistribute when servers are added or removed</p>
        </CardHeader>
        <CardContent>
          {/* Explanation */}
          <div className="bg-blue-50 border-l-4 border-blue-600 p-4 rounded-r-lg mb-6">
            <h3 className="font-semibold text-blue-900 mb-2">How It Works</h3>
            <p className="text-sm text-gray-700 mb-2">
              <strong className="text-blue-800">The Circle (Hash Ring):</strong> Represents all possible hash values from 0° to 360°. Both servers and data keys are mapped onto this ring.
            </p>
            <p className="text-sm text-gray-700 mb-2">
              <strong className="text-blue-800">Virtual Nodes:</strong> Each physical server has multiple positions on the ring for even distribution. Same-colored nodes = same physical server.
            </p>
            <p className="text-sm text-gray-700">
              <strong className="text-blue-800">Data Keys (Small dots):</strong> Assigned to the first server found by moving clockwise around the ring.
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap gap-3 mb-6">
            <button
              onClick={addServer}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center gap-2"
            >
              <Plus size={18} />
              Add Server
            </button>
            <button
              onClick={removeServer}
              disabled={uniqueServerCount <= 1}
              className="px-4 py-2 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Minus size={18} />
              Remove Server
            </button>
            <button
              onClick={addKey}
              className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center gap-2"
            >
              <Plus size={18} />
              Add Key
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors flex items-center gap-2"
            >
              <RotateCcw size={18} />
              Reset
            </button>
            <div className="flex flex-col gap-1 px-4 py-2 bg-gray-100 rounded-lg min-w-[220px]">
              <label className="text-sm font-medium text-gray-700">
                Virtual Nodes per Server: <span className="text-blue-600 font-bold">{virtualNodes}</span>
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={virtualNodes}
                onChange={(e) => updateVirtualNodes(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Canvas */}
          <div className="flex justify-center mb-6">
            <canvas
              ref={canvasRef}
              width={700}
              height={700}
              className="border border-gray-200 rounded-lg bg-white"
            />
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Physical Servers</div>
              <div className="text-2xl font-bold text-gray-900">{uniqueServerCount}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Virtual Nodes (Total)</div>
              <div className="text-2xl font-bold text-gray-900">{servers.length}</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="text-xs text-gray-500 uppercase font-semibold mb-1">Data Keys</div>
              <div className="text-2xl font-bold text-gray-900">{keys.length}</div>
            </div>
            <div className="bg-gradient-to-br from-amber-100 to-amber-200 p-4 rounded-lg border-2 border-amber-400">
              <div className="text-xs text-amber-800 uppercase font-semibold mb-1">Keys Moved</div>
              <div className="text-2xl font-bold text-gray-900">{lastAction ? lastMovedKeys : '-'}</div>
              {lastAction && <div className="text-xs text-amber-700 mt-1 font-medium">{lastAction}</div>}
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-6 mb-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-blue-500 border-2 border-blue-800"></div>
              <span className="text-sm text-gray-600"><strong>Virtual Node</strong> - Same color = same physical server</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-emerald-500 border-2 border-emerald-700"></div>
              <span className="text-sm text-gray-600"><strong>Data Key</strong> - Assigned to next clockwise server</span>
            </div>
          </div>

          {/* Demo Tips */}
          <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r-lg">
            <h3 className="font-semibold text-amber-900 mb-2">Try This Demo:</h3>
            <ol className="list-decimal list-inside space-y-2 text-sm text-gray-700">
              <li><strong className="text-amber-800">Adjust Virtual Nodes:</strong> More virtual nodes = better load balancing!</li>
              <li><strong className="text-amber-800">Remove Server:</strong> Watch the "Keys Moved" counter - only affected keys relocate</li>
              <li><strong className="text-amber-800">Add Server:</strong> See how it takes over part of other servers' ranges</li>
              <li><strong className="text-amber-800">Compare:</strong> Traditional hashing (hash % N) would reassign nearly ALL keys when servers change!</li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
