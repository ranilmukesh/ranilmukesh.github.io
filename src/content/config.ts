import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        description: z.string(),
        pubDate: z.coerce.date(),
        updatedDate: z.coerce.date().optional(),
        heroImage: z.string().optional(),
    }),
});

const research = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        venue: z.string().optional(),
        date: z.coerce.date().optional(),
        pdfUrl: z.string(),
        githubUrl: z.string().optional(),
        abstract: z.string().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

export const collections = { blog, research };
