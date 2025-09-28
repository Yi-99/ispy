import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_PUBLIC_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_PUBLIC_SUPABASE_ANON_KEY;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Storage bucket name for image uploads
export const STORAGE_BUCKET = import.meta.env.VITE_STORAGE_NAME;
export const UPLOAD_FOLDER = import.meta.env.VITE_UPLOAD_FOLDER;
export const FRAUD_FOLDER = import.meta.env.VITE_FRAUD_FOLDER;
export const NON_FRAUD_FOLDER = import.meta.env.VITE_NON_FRAUD_FOLDER;
