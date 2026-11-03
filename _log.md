#### 2.11.2026.
#### Audio timeline fix
##### html5 player needs to be refreshed a few times for player bar to work properly.

Cause:  

```python
-m http.server
```
 has no Range support, so the player bar needed 1-2 refreshes.

<p> fix: server with Range requests support: </p>

```js
npx http-server -p 8000 -c-1
```

note: Hosting on GitHub Pages: supports Range requests, so this problem shouldn’t reappear.