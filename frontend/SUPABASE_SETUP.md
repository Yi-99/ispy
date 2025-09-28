# Supabase Setup Instructions

This application uses Supabase for image storage and management. Follow these steps to set up Supabase for the iSpy application.

## 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up/sign in
2. Click "New Project"
3. Choose your organization and enter project details:
   - Name: `ispy-fraud-detection`
   - Database Password: (generate a strong password)
   - Region: Choose the closest region to your users
4. Click "Create new project"

## 2. Get Your Project Credentials

1. In your Supabase dashboard, go to Settings > API
2. Copy the following values:
   - **Project URL** (starts with `https://`)
   - **anon public** key (starts with `eyJ`)

## 3. Create Storage Bucket

1. In your Supabase dashboard, go to Storage
2. Click "Create a new bucket"
3. Name: `vehicle-damage-images`
4. Make it **Public** (so images can be accessed via URL)
5. Click "Create bucket"

## 4. Set Up Environment Variables

Create a `.env.local` file in the `frontend` directory with your Supabase credentials:

```env
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

Replace the values with your actual Supabase project URL and anon key.

## 5. Configure Storage Policies (Optional but Recommended)

For production use, you should set up Row Level Security (RLS) policies:

1. Go to Storage > Policies in your Supabase dashboard
2. Create policies for the `vehicle-damage-images` bucket:

**Policy 1: Allow public read access**
```sql
CREATE POLICY "Public read access" ON storage.objects
FOR SELECT USING (bucket_id = 'vehicle-damage-images');
```

**Policy 2: Allow authenticated users to upload**
```sql
CREATE POLICY "Allow uploads" ON storage.objects
FOR INSERT WITH CHECK (bucket_id = 'vehicle-damage-images');
```

**Policy 3: Allow users to delete their own files**
```sql
CREATE POLICY "Allow deletes" ON storage.objects
FOR DELETE USING (bucket_id = 'vehicle-damage-images');
```

## 6. Test the Setup

1. Start the development server: `npm run dev`
2. Navigate to the Upload & Analyze page
3. Try uploading an image to test the connection

## Troubleshooting

### Common Issues:

1. **"Invalid API key" error**: Check that your environment variables are correct
2. **"Bucket not found" error**: Make sure you created the `vehicle-damage-images` bucket
3. **CORS errors**: Ensure your Supabase project allows requests from your domain
4. **Upload fails**: Check that the storage bucket is set to public or has proper RLS policies

### Environment Variables Not Loading:

Make sure your `.env.local` file is in the `frontend` directory and restart the development server.

## Security Notes

- The anon key is safe to use in client-side code
- For production, consider implementing user authentication
- Set up proper RLS policies to control access to uploaded images
- Consider implementing file size and type restrictions
