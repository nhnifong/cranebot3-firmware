// See src/auth-bridge.ts and README.md for why this has to be a separate
// module script from app.ts, loaded first.
import '../src/playroom-styles.css';
import '../src/mobile.css';
import appShellHtml from '../src/app-shell.html?raw';
import { stubAuthBridge } from './stub-bridge.ts';

const resolvedShellHtml = appShellHtml.replaceAll(
  '%VITE_ASSET_BUCKET_URL%',
  import.meta.env.VITE_ASSET_BUCKET_URL ?? ''
);
document.body.insertAdjacentHTML('afterbegin', resolvedShellHtml);

window.__playroomAuthBridge = stubAuthBridge;
