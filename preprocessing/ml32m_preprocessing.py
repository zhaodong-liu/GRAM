#!/usr/bin/env python3
"""
MovieLens-32M Data Preprocessing Script for GRAM
Converts MovieLens-32M format to GRAM-compatible format

Author: AI Assistant
Date: 2025
Usage: python ml32m_preprocessing.py --input_dir /path/to/ml-32m --output_dir ../rec_datasets/ML32M
"""

import pandas as pd
import json
import numpy as np
from collections import defaultdict, Counter
import os
import argparse
from pathlib import Path
import logging
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml32m_preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MovieLensPreprocessor:
    """MovieLens-32M preprocessor for GRAM compatibility"""
    
    def __init__(self, input_dir, output_dir, min_interactions=5):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_interactions = min_interactions
        
        # Data containers
        self.ratings = None
        self.movies = None
        self.tags = None
        self.links = None
        
        # Processed data
        self.filtered_ratings = None
        self.filtered_movies = None
        self.user_sequences = {}
        self.item_metadata = {}
        
        # Statistics
        self.stats = {
            'original': {},
            'filtered': {},
            'processing_time': {}
        }
    
    def load_data(self):
        """Load MovieLens-32M data files"""
        logger.info("🎬 Loading MovieLens-32M data...")
        start_time = datetime.now()
        
        try:
            # Load ratings
            ratings_file = self.input_dir / 'ratings.csv'
            if not ratings_file.exists():
                raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
            
            logger.info("📊 Loading ratings...")
            self.ratings = pd.read_csv(ratings_file)
            logger.info(f"   Loaded {len(self.ratings):,} ratings")
            
            # Load movies metadata
            movies_file = self.input_dir / 'movies.csv'
            if not movies_file.exists():
                raise FileNotFoundError(f"Movies file not found: {movies_file}")
            
            logger.info("🎭 Loading movies...")
            self.movies = pd.read_csv(movies_file)
            logger.info(f"   Loaded {len(self.movies):,} movies")
            
            # Load tags (optional)
            tags_file = self.input_dir / 'tags.csv'
            if tags_file.exists():
                logger.info("🏷️  Loading tags...")
                self.tags = pd.read_csv(tags_file)
                logger.info(f"   Loaded {len(self.tags):,} tags")
            else:
                logger.warning("⚠️  Tags file not found, proceeding without tags")
                self.tags = None
            
            # Load links (optional)
            links_file = self.input_dir / 'links.csv'
            if links_file.exists():
                logger.info("🔗 Loading links...")
                self.links = pd.read_csv(links_file)
                logger.info(f"   Loaded {len(self.links):,} links")
            else:
                logger.warning("⚠️  Links file not found, proceeding without links")
                self.links = None
            
            # Store original statistics
            self.stats['original'] = {
                'num_ratings': len(self.ratings),
                'num_users': self.ratings['userId'].nunique(),
                'num_movies': self.ratings['movieId'].nunique(),
                'num_movies_with_metadata': len(self.movies),
                'avg_ratings_per_user': len(self.ratings) / self.ratings['userId'].nunique(),
                'avg_ratings_per_movie': len(self.ratings) / self.ratings['movieId'].nunique(),
                'rating_sparsity': 1.0 - (len(self.ratings) / (self.ratings['userId'].nunique() * self.ratings['movieId'].nunique()))
            }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_time']['loading'] = processing_time
            logger.info(f"✅ Data loading completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def clean_and_validate_data(self):
        """Clean and validate the loaded data"""
        logger.info("🧹 Cleaning and validating data...")
        start_time = datetime.now()
        
        # Remove invalid ratings
        original_count = len(self.ratings)
        
        # Remove NaN values
        self.ratings = self.ratings.dropna()
        logger.info(f"   Removed {original_count - len(self.ratings):,} NaN ratings")
        
        # Ensure positive IDs
        self.ratings = self.ratings[
            (self.ratings['userId'] > 0) & 
            (self.ratings['movieId'] > 0) &
            (self.ratings['rating'] > 0)
        ]
        
        # Sort by user and timestamp
        self.ratings = self.ratings.sort_values(['userId', 'timestamp'])
        
        # Validate movies data
        self.movies = self.movies.dropna(subset=['movieId', 'title'])
        self.movies = self.movies[self.movies['movieId'] > 0]
        
        # Clean movie titles and genres
        self.movies['title'] = self.movies['title'].str.strip()
        self.movies['genres'] = self.movies['genres'].fillna('(no genres listed)')
        
        logger.info(f"✅ Data cleaning completed")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['cleaning'] = processing_time
    
    def apply_filtering(self):
        """Apply k-core filtering"""
        logger.info(f"🔍 Applying {self.min_interactions}-core filtering...")
        start_time = datetime.now()
        
        ratings = self.ratings.copy()
        movies = self.movies.copy()
        
        iteration = 0
        while True:
            iteration += 1
            logger.info(f"   Iteration {iteration}...")
            
            # Count interactions
            user_counts = ratings['userId'].value_counts()
            movie_counts = ratings['movieId'].value_counts()
            
            # Filter users with sufficient interactions
            valid_users = user_counts[user_counts >= self.min_interactions].index
            ratings = ratings[ratings['userId'].isin(valid_users)]
            
            # Filter movies with sufficient interactions
            valid_movies = movie_counts[movie_counts >= self.min_interactions].index
            ratings = ratings[ratings['movieId'].isin(valid_movies)]
            
            # Check convergence
            new_user_counts = ratings['userId'].value_counts()
            new_movie_counts = ratings['movieId'].value_counts()
            
            users_converged = (new_user_counts >= self.min_interactions).all()
            movies_converged = (new_movie_counts >= self.min_interactions).all()
            
            logger.info(f"     Users: {len(new_user_counts):,}, Movies: {len(new_movie_counts):,}, Ratings: {len(ratings):,}")
            
            if users_converged and movies_converged:
                logger.info(f"   ✅ Converged after {iteration} iterations")
                break
            
            if iteration > 20:  # Safety check
                logger.warning(f"   ⚠️  Stopping after {iteration} iterations (safety limit)")
                break
        
        # Update data
        self.filtered_ratings = ratings
        
        # Filter movies metadata to only include remaining movies
        remaining_movie_ids = ratings['movieId'].unique()
        self.filtered_movies = movies[movies['movieId'].isin(remaining_movie_ids)]
        
        # Store filtered statistics
        self.stats['filtered'] = {
            'num_ratings': len(self.filtered_ratings),
            'num_users': self.filtered_ratings['userId'].nunique(),
            'num_movies': self.filtered_ratings['movieId'].nunique(),
            'num_movies_with_metadata': len(self.filtered_movies),
            'avg_ratings_per_user': len(self.filtered_ratings) / self.filtered_ratings['userId'].nunique(),
            'avg_ratings_per_movie': len(self.filtered_ratings) / self.filtered_ratings['movieId'].nunique(),
            'data_retention_rate': len(self.filtered_ratings) / len(self.ratings)
        }
        
        logger.info(f"📈 Filtering results:")
        logger.info(f"   Users: {self.stats['original']['num_users']:,} → {self.stats['filtered']['num_users']:,}")
        logger.info(f"   Movies: {self.stats['original']['num_movies']:,} → {self.stats['filtered']['num_movies']:,}")
        logger.info(f"   Ratings: {self.stats['original']['num_ratings']:,} → {self.stats['filtered']['num_ratings']:,}")
        logger.info(f"   Retention rate: {self.stats['filtered']['data_retention_rate']:.2%}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['filtering'] = processing_time
    
    def extract_movie_features(self):
        """Extract and process movie features"""
        logger.info("🎭 Extracting movie features...")
        start_time = datetime.now()
        
        # Process each movie
        for _, movie in self.filtered_movies.iterrows():
            movie_id = movie['movieId']
            title = movie['title']
            genres = movie['genres']
            
            # Extract year from title
            year = self._extract_year_from_title(title)
            clean_title = self._clean_title(title)
            
            # Process genres
            genre_list = self._process_genres(genres)
            
            # Get tags if available
            movie_tags = self._get_movie_tags(movie_id) if self.tags is not None else []
            
            # Create metadata
            self.item_metadata[str(movie_id)] = {
                'title': clean_title,
                'original_title': title,
                'genres': genre_list,
                'year': year,
                'tags': movie_tags,
                'text': self._create_movie_text(clean_title, genre_list, year, movie_tags)
            }
        
        logger.info(f"✅ Extracted features for {len(self.item_metadata):,} movies")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['feature_extraction'] = processing_time
    
    def _extract_year_from_title(self, title):
        """Extract year from movie title"""
        # Look for year in parentheses at the end
        year_match = re.search(r'\((\d{4})\)$', title)
        if year_match:
            return year_match.group(1)
        return ""
    
    def _clean_title(self, title):
        """Clean movie title by removing year"""
        # Remove year in parentheses
        clean_title = re.sub(r'\s*\(\d{4}\)$', '', title)
        return clean_title.strip()
    
    def _process_genres(self, genres_str):
        """Process genre string into list"""
        if pd.isna(genres_str) or genres_str == "(no genres listed)":
            return []
        
        # Split by | and clean
        genres = [genre.strip() for genre in genres_str.split('|') if genre.strip()]
        return genres
    
    def _get_movie_tags(self, movie_id):
        """Get tags for a specific movie"""
        if self.tags is None:
            return []
        
        movie_tags = self.tags[self.tags['movieId'] == movie_id]['tag'].tolist()
        
        # Get most common tags (limit to top 5)
        if movie_tags:
            tag_counts = Counter(movie_tags)
            top_tags = [tag for tag, count in tag_counts.most_common(5)]
            return top_tags
        
        return []
    
    def _create_movie_text(self, title, genres, year, tags):
        """Create comprehensive text representation of movie"""
        text_parts = [title]
        
        if genres:
            text_parts.append(f"Genres: {', '.join(genres)}")
        
        if year:
            text_parts.append(f"Year: {year}")
        
        if tags:
            text_parts.append(f"Tags: {', '.join(tags)}")
        
        return " | ".join(text_parts)
    
    def create_user_sequences(self):
        """Create user interaction sequences"""
        logger.info("👥 Creating user sequences...")
        start_time = datetime.now()
        
        # Group by user and create sequences
        user_groups = self.filtered_ratings.groupby('userId')
        
        for user_id, user_ratings in user_groups:
            # Sort by timestamp to ensure chronological order
            user_ratings = user_ratings.sort_values('timestamp')
            
            # Convert to list of movie IDs
            sequence = user_ratings['movieId'].astype(str).tolist()
            self.user_sequences[str(user_id)] = sequence
        
        # Calculate sequence statistics
        sequence_lengths = [len(seq) for seq in self.user_sequences.values()]
        
        self.stats['sequences'] = {
            'num_users': len(self.user_sequences),
            'total_interactions': sum(sequence_lengths),
            'avg_sequence_length': np.mean(sequence_lengths),
            'median_sequence_length': np.median(sequence_lengths),
            'min_sequence_length': min(sequence_lengths),
            'max_sequence_length': max(sequence_lengths),
            'std_sequence_length': np.std(sequence_lengths)
        }
        
        logger.info(f"✅ Created sequences for {len(self.user_sequences):,} users")
        logger.info(f"   Average sequence length: {self.stats['sequences']['avg_sequence_length']:.1f}")
        logger.info(f"   Median sequence length: {self.stats['sequences']['median_sequence_length']:.1f}")
        logger.info(f"   Range: {self.stats['sequences']['min_sequence_length']} - {self.stats['sequences']['max_sequence_length']}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['sequence_creation'] = processing_time
    
    def save_gram_format(self):
        """Save data in GRAM-compatible format"""
        logger.info(f"💾 Saving data in GRAM format to {self.output_dir}...")
        start_time = datetime.now()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save user sequences
        sequences_file = self.output_dir / "ML32M_sequences.json"
        logger.info(f"   Saving user sequences to {sequences_file}...")
        with open(sequences_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_sequences, f, indent=2, ensure_ascii=False)
        
        # Save item metadata
        metadata_file = self.output_dir / "ML32M_item_metadata.json"
        logger.info(f"   Saving item metadata to {metadata_file}...")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.item_metadata, f, indent=2, ensure_ascii=False)
        
        # Create dataset info
        dataset_info = {
            'dataset_name': 'ML32M',
            'description': 'MovieLens 32M dataset processed for GRAM',
            'preprocessing_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'format_version': '1.0',
            'min_interactions_filter': self.min_interactions,
            'data_files': {
                'user_sequences': 'ML32M_sequences.json',
                'item_metadata': 'ML32M_item_metadata.json',
                'dataset_info': 'ML32M_info.json'
            }
        }
        
        info_file = self.output_dir / "ML32M_info.json"
        logger.info(f"   Saving dataset info to {info_file}...")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # Create GRAM-style text files for compatibility
        self._create_gram_text_files()
        
        logger.info(f"✅ Data saved successfully!")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats['processing_time']['saving'] = processing_time
    
    def _create_gram_text_files(self):
        """Create GRAM-style text files for compatibility"""
        logger.info("📝 Creating GRAM-compatible text files...")
        
        # Create user_sequence.txt
        user_seq_file = self.output_dir / "user_sequence.txt"
        with open(user_seq_file, 'w') as f:
            for user_id, sequence in self.user_sequences.items():
                f.write(f"{user_id} {' '.join(sequence)}\n")
        
        # Create item_plain_text.txt
        item_text_file = self.output_dir / "item_plain_text.txt"
        with open(item_text_file, 'w') as f:
            for item_id, metadata in self.item_metadata.items():
                text = metadata['text'].replace('\n', ' ').strip()
                f.write(f"{item_id} {text}\n")
        
        logger.info("   ✅ Created user_sequence.txt")
        logger.info("   ✅ Created item_plain_text.txt")
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        logger.info("📊 Dataset Statistics Summary")
        logger.info("=" * 50)
        
        # Original vs Filtered
        logger.info("🔢 Data Size:")
        logger.info(f"   Original - Users: {self.stats['original']['num_users']:,}, "
                   f"Movies: {self.stats['original']['num_movies']:,}, "
                   f"Ratings: {self.stats['original']['num_ratings']:,}")
        logger.info(f"   Filtered - Users: {self.stats['filtered']['num_users']:,}, "
                   f"Movies: {self.stats['filtered']['num_movies']:,}, "
                   f"Ratings: {self.stats['filtered']['num_ratings']:,}")
        logger.info(f"   Retention: {self.stats['filtered']['data_retention_rate']:.1%}")
        
        # Sequence Statistics
        if 'sequences' in self.stats:
            logger.info("📏 Sequence Statistics:")
            logger.info(f"   Average length: {self.stats['sequences']['avg_sequence_length']:.1f}")
            logger.info(f"   Median length: {self.stats['sequences']['median_sequence_length']:.1f}")
            logger.info(f"   Min length: {self.stats['sequences']['min_sequence_length']}")
            logger.info(f"   Max length: {self.stats['sequences']['max_sequence_length']}")
            logger.info(f"   Std deviation: {self.stats['sequences']['std_sequence_length']:.1f}")
        
        # Genre Statistics
        if self.item_metadata:
            all_genres = []
            for metadata in self.item_metadata.values():
                all_genres.extend(metadata['genres'])
            
            genre_counts = Counter(all_genres)
            logger.info("🎭 Top 10 Genres:")
            for genre, count in genre_counts.most_common(10):
                logger.info(f"   {genre}: {count:,} ({count/len(self.item_metadata)*100:.1f}%)")
        
        # Processing Time
        total_time = sum(self.stats['processing_time'].values())
        logger.info("⏱️  Processing Time:")
        for step, time_taken in self.stats['processing_time'].items():
            logger.info(f"   {step.replace('_', ' ').title()}: {time_taken:.2f}s ({time_taken/total_time*100:.1f}%)")
        logger.info(f"   Total: {total_time:.2f}s")
        
        logger.info("=" * 50)
    
    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        logger.info("🚀 Starting MovieLens-32M preprocessing for GRAM...")
        total_start_time = datetime.now()
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Clean and validate
            self.clean_and_validate_data()
            
            # Step 3: Apply filtering
            self.apply_filtering()
            
            # Step 4: Extract features
            self.extract_movie_features()
            
            # Step 5: Create sequences
            self.create_user_sequences()
            
            # Step 6: Save results
            self.save_gram_format()
            
            # Step 7: Print statistics
            self.print_statistics()
            
            total_time = (datetime.now() - total_start_time).total_seconds()
            logger.info(f"🎉 Preprocessing completed successfully in {total_time:.2f} seconds!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Preprocess MovieLens-32M for GRAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml32m_preprocessing.py --input_dir /data/ml-32m --output_dir ../rec_datasets/ML32M
  python ml32m_preprocessing.py --input_dir ./ml-32m --min_interactions 10 --output_dir ./processed_data
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Path to MovieLens-32M data directory (containing ratings.csv, movies.csv, etc.)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='../rec_datasets/ML32M',
        help='Output directory for processed data (default: ../rec_datasets/ML32M)'
    )
    
    parser.add_argument(
        '--min_interactions', 
        type=int, 
        default=5,
        help='Minimum interactions for users and items (k-core filtering, default: 5)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--no_tags',
        action='store_true',
        help='Skip loading tags even if tags.csv exists'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print header
    logger.info("🎬 MovieLens-32M to GRAM Preprocessor")
    logger.info("=" * 50)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Min interactions: {args.min_interactions}")
    logger.info(f"Skip tags: {args.no_tags}")
    logger.info("=" * 50)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"❌ Input directory does not exist: {input_path}")
        return 1
    
    required_files = ['ratings.csv', 'movies.csv']
    missing_files = [f for f in required_files if not (input_path / f).exists()]
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        logger.error("💡 Please ensure you have the complete MovieLens-32M dataset")
        return 1
    
    # Create preprocessor and run
    preprocessor = MovieLensPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_interactions=args.min_interactions
    )
    
    # Override tags loading if requested
    if args.no_tags:
        logger.info("⚠️  Skipping tags as requested")
    
    # Run preprocessing
    success = preprocessor.run_full_preprocessing()
    
    if success:
        logger.info("🎉 All done! You can now run GRAM training with:")
        logger.info(f"   cd GRAM/command")
        logger.info(f"   bash train_gram_unified.sh 2  # Select MovieLens")
        logger.info(f"   # Then choose ML32M")
        return 0
    else:
        logger.error("❌ Preprocessing failed!")
        return 1


if __name__ == "__main__":
    exit(main())